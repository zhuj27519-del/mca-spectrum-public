import csv
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from protocol import Frame, StreamParser, encode, decode
from simulation import simulate
from analysis import analyze

def test_protocol_fragment_coalesce_resync():
    packets = [encode(simulate(1000+i*100, i)) for i in range(20)]
    bad = bytearray(packets[0]); bad[-1] ^= 1
    malformed = bytearray(packets[0]); malformed[24:28] = b'\xff'*4
    stream = b'garbage'+bytes(malformed)+bytes(bad)+b''.join(packets)
    parser, frames = StreamParser(), []
    rng = random.Random(743)
    while stream:
        n = rng.randint(1, 17000)
        frames.extend(parser.feed(stream[:n])); stream = stream[n:]
        assert len(parser.buffer) < 8224
    assert [f.sequence for f in frames] == list(range(20))
    assert parser.crc_errors >= 1 and parser.header_errors >= 1
    assert not parser.buffer

@pytest.mark.parametrize('field', [0, 4, 6, 12, 20, 22, 24, 28, 500, 8223])
def test_invalid_frame(field):
    bad = bytearray(encode(simulate()))
    bad[field] ^= 0xff
    with pytest.raises(ValueError):
        decode(bad)

def test_extremes():
    frame = Frame(0xffffffff, 2**64-1, np.full(2048, 0xffffffff, np.uint32), True)
    actual = decode(encode(frame))
    assert actual.total == 2048*0xffffffff and actual.saturated
    assert actual.duration_ms == 2**64-1 and np.isfinite(actual.cps)
    with pytest.raises(ValueError):
        encode(Frame(0, 0, np.zeros(2048, np.uint32)))
    parser = StreamParser()
    assert not list(parser.feed(b'X'*1000000)) and len(parser.buffer) == 3

def test_analysis_phase2_golden():
    frame = simulate(1000)
    peaks, _ = analyze(frame.spectrum)
    assert frame.total == 124897
    assert len(peaks) == 3
    assert np.allclose([p.centroid for p in peaks], [400.136480771,1000.197142179,1599.874416807], atol=1e-4)
    assert np.allclose([p.fwhm for p in peaks], [23.703878447,37.905353073,49.870909169], atol=2e-4)
    peaks, _ = analyze(np.zeros(2048))
    assert not peaks

@pytest.mark.parametrize('duration', [500, 1000, 2078, 10000, 60000])
@pytest.mark.parametrize('noise', [0, 1, 2])
def test_simulation_matrix(duration, noise):
    frame = simulate(duration, noise=noise)
    peaks, _ = analyze(frame.spectrum)
    assert len(peaks) == 3
    assert np.allclose([p.centroid for p in peaks], [400,1000,1600], atol=2.5)

@pytest.fixture(scope='module')
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])

def wait_for(app, predicate, timeout=5):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Qt condition timed out')

def test_gui_simulation_csv_log(app, tmp_path):
    from viewer import Viewer
    window = Viewer(simulation=True)
    window.show()
    try:
        wait_for(app, lambda: window.current is not None)
        assert len(window.curve.getData()[0]) == 2048
        assert window.table.rowCount() == 3
        assert window.plot.getAxis('bottom').labelText == 'Channel'
        window.log.setChecked(True); app.processEvents()
        assert window.plot.getPlotItem().ctrl.logYCheck.isChecked()
        window.autoscale.setChecked(False)
        window.plot.setXRange(300, 500)
        path = tmp_path/'spectrum.csv'; window.save_csv(path)
        rows = list(csv.reader(path.open()))
        assert len(rows) == 2050 and rows[1] == ['channel', 'counts'] and rows[-1][0] == '2047'
        window.log.setChecked(False); window.autoscale.setChecked(True)
        app.processEvents()
        screenshot = os.environ.get('MCA_SCREENSHOT')
        if screenshot:
            assert window.grab().save(screenshot)
    finally:
        window.close()

def test_gui_tcp_reconnect(app):
    from PySide6.QtNetwork import QNetworkProxy
    previous_proxy = QNetworkProxy.applicationProxy()
    QNetworkProxy.setApplicationProxy(QNetworkProxy(QNetworkProxy.ProxyType.HttpProxy, '127.0.0.1', 9))
    from viewer import Viewer
    from mock_server import MockServer
    server = MockServer(port=0, interval=.03, disconnect_every=4).start()
    window = Viewer(simulation=False, host='127.0.0.1', port=server.port)
    try:
        wait_for(app, lambda: window.receiver.received >= 12, timeout=10)
        wait_for(app, lambda: window.current is not None)
        assert window.receiver.connections >= 3 and server.connections >= 3
        assert window.receiver.parser.crc_errors == 0
        assert len(window.current.spectrum) == 2048
        window.stop(); old = window.receiver.connections
        end = time.monotonic()+.8
        while time.monotonic()<end:
            app.processEvents(); time.sleep(.01)
        assert window.receiver.connections == old
    finally:
        window.close(); server.close()
        QNetworkProxy.setApplicationProxy(previous_proxy)
