"""Run: python viewer.py --simulate, or --host 127.0.0.1 --port 5000."""
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtNetwork, QtWidgets
import pyqtgraph as pg
from protocol import StreamParser, encode
from simulation import simulate
from analysis import analyze

class Receiver(QtCore.QObject):
    frame_received = QtCore.Signal(object)
    status_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.socket = QtNetwork.QTcpSocket(self)
        # This is a direct instrument TCP connection. System HTTP proxies cannot
        # tunnel arbitrary raw sockets and may reject LAN targets on Windows.
        self.socket.setProxy(QtNetwork.QNetworkProxy(QtNetwork.QNetworkProxy.ProxyType.NoProxy))
        self.socket.setReadBufferSize(65536)
        self.socket.readyRead.connect(self.read)
        self.socket.connected.connect(self.connected)
        self.socket.disconnected.connect(self.retry)
        self.socket.errorOccurred.connect(lambda _: self.retry())
        self.retry_timer = QtCore.QTimer(self)
        self.retry_timer.setSingleShot(True)
        self.retry_timer.timeout.connect(self.connect_now)
        self.connect_timeout = QtCore.QTimer(self)
        self.connect_timeout.setSingleShot(True)
        self.connect_timeout.timeout.connect(self.expired)
        self.enabled = False
        self.parser = StreamParser()
        self.connections = self.received = self.errors = 0
        self.delay = 500

    def start(self, host, port):
        self.stop()
        self.host, self.port = host, port
        self.enabled = True
        self.delay = 500
        self.connect_now()

    def stop(self):
        self.enabled = False
        self.retry_timer.stop()
        self.connect_timeout.stop()
        self.socket.abort()
        self.parser.buffer.clear()
        self.status_changed.emit('Stopped — displayed data may be stale')

    def connect_now(self):
        if not self.enabled:
            return
        self.parser.buffer.clear()  # Never combine bytes from different sessions.
        self.socket.abort()
        self.status_changed.emit(f'Connecting to {self.host}:{self.port}…')
        self.socket.connectToHost(self.host, self.port)
        self.connect_timeout.start(3000)

    def connected(self):
        self.retry_timer.stop()
        self.connect_timeout.stop()
        self.delay = 500
        self.connections += 1
        self.status_changed.emit(f'Connected to {self.host}:{self.port}')

    def expired(self):
        if self.enabled:
            self.socket.abort()
            self.retry()

    def retry(self):
        self.connect_timeout.stop()
        if self.enabled and not self.retry_timer.isActive():
            self.errors += 1
            self.status_changed.emit(f'Reconnecting — {self.socket.errorString()} (data may be stale)')
            self.retry_timer.start(self.delay)
            self.delay = min(5000, self.delay * 2)

    def read(self):
        if not self.enabled:
            return
        data = bytes(self.socket.read(65536))
        for frame in self.parser.feed(data):
            self.received += 1
            self.frame_received.emit(frame)
        if self.socket.bytesAvailable():
            QtCore.QTimer.singleShot(0, self.read)

class Viewer(QtWidgets.QMainWindow):
    def __init__(self, simulation=True, host='127.0.0.1', port=5000, autostart=True):
        super().__init__()
        # Qt's offscreen platform may have no system font enumeration on Windows.
        font_path = Path('C:/Windows/Fonts/segoeui.ttf')
        if font_path.exists():
            QtGui.QFontDatabase.addApplicationFont(str(font_path))
        QtWidgets.QApplication.instance().setFont(QtGui.QFont('Segoe UI', 10))
        self.setWindowTitle('Apollo MCA · 2048-channel spectrum')
        self.resize(1180, 790)
        pg.setConfigOptions(antialias=True, background='#101b2b', foreground='#dce8f5')
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        title = QtWidgets.QLabel('APOLLO MCA    /    REAL-TIME SPECTRUM')
        title.setStyleSheet('font-size:22px;font-weight:600;padding:12px;color:#103655')
        layout.addWidget(title)
        controls = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Local simulation', 'TCP acquisition'])
        self.mode.setCurrentIndex(0 if simulation else 1)
        self.host = QtWidgets.QLineEdit(host)
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(port)
        self.start_button = QtWidgets.QPushButton('Start')
        self.stop_button = QtWidgets.QPushButton('Stop')
        self.log = QtWidgets.QCheckBox('Log Y')
        self.autoscale = QtWidgets.QCheckBox('Auto range')
        self.autoscale.setChecked(True)
        self.save_button = QtWidgets.QPushButton('Save CSV…')
        for widget in [self.mode, self.host, self.port, self.start_button, self.stop_button, self.log, self.autoscale, self.save_button]:
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.metrics = QtWidgets.QLabel('Awaiting validated spectrum…')
        self.metrics.setStyleSheet('font-size:17px;padding:10px')
        layout.addWidget(self.metrics)
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Channel')
        self.plot.setLabel('left', 'Counts / channel')
        self.plot.showGrid(x=True, y=True, alpha=.15)
        self.curve = self.plot.plot(pen=pg.mkPen('#54dcc9', width=1.8), fillLevel=0, brush=(40, 145, 153, 35))
        self.plot.setXRange(0, 2047)
        self.plot.getViewBox().sigRangeChangedManually.connect(lambda _: self.autoscale.setChecked(False))
        layout.addWidget(self.plot, 1)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Peak', 'Centroid / channel', 'FWHM / channels', 'Channel resolution / %'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(165)
        layout.addWidget(self.table)
        self.note = QtWidgets.QLabel('Uncalibrated • Channel axis • Peak analysis recomputed on PC from MCA1 counts; no keV calibration')
        layout.addWidget(self.note)
        self.connection = QtWidgets.QLabel('Stopped')
        self.health = QtWidgets.QLabel('')
        layout.addWidget(self.connection)
        layout.addWidget(self.health)
        self.receiver = Receiver(self)
        self.receiver.frame_received.connect(self.receive)
        self.receiver.status_changed.connect(self.connection.setText)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.save_button.clicked.connect(self.save_dialog)
        self.log.toggled.connect(self.set_log)
        self.autoscale.toggled.connect(lambda enabled: self.plot.enableAutoRange(enable=enabled))
        self.pending = self.current = None
        self.last_frame_time = 0.0
        self.annotations = []
        self.sim_count = 0
        self.running_sim = False
        self.sim_parser = StreamParser()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)
        if autostart:
            self.start()

    def start(self):
        self.stop()
        self.pending = None
        self.sim_count = 0
        self.running_sim = self.mode.currentIndex() == 0
        if self.running_sim:
            self.connection.setText('LOCAL SIMULATION — deterministic test data, no hardware connection')
        else:
            self.receiver.start(self.host.text().strip(), self.port.value())

    def stop(self):
        self.running_sim = False
        self.receiver.stop()

    def receive(self, frame):
        self.pending = frame  # Coalesce display updates, never grow a GUI queue.
        self.last_frame_time = time.monotonic()

    def update_frame(self):
        if self.running_sim:
            self.sim_count += 1
            for frame in self.sim_parser.feed(encode(simulate(1000+100*self.sim_count, self.sim_count))):
                self.receive(frame)
        if self.pending is not None:
            self.current, self.pending = self.pending, None
            self.render(self.current)
        parser = self.sim_parser if self.running_sim else self.receiver.parser
        age = time.monotonic()-self.last_frame_time if self.last_frame_time else 0
        self.health.setText(f'Valid frames: {parser.frames:,}    CRC rejects: {parser.crc_errors}    Header rejects: {parser.header_errors}    '
                            f'Connections: {self.receiver.connections}    Frame age: {age:.1f} s' + ('    STALE' if age > 2 else ''))

    def set_log(self, enabled):
        self.plot.setLogMode(y=enabled)
        if self.current is not None:
            self.render(self.current)

    def render(self, frame):
        peaks, _ = analyze(frame.spectrum)
        self.curve.setData(np.arange(2048), frame.spectrum.astype(float))
        self.metrics.setText(f'CPS  {frame.cps:,.2f}       Total  {frame.total:,}       Exposure  {frame.duration_ms/1000:.3f} s'
                             f'       Peaks  {len(peaks)}       Seq  {frame.sequence}' + ('    SATURATED' if frame.saturated else ''))
        for item in self.annotations:
            self.plot.removeItem(item)
        self.annotations.clear()
        self.table.setRowCount(len(peaks))
        for k, peak in enumerate(peaks):
            values = [str(k+1), f'{peak.centroid:.4f}', f'{peak.fwhm:.4f}' if peak.fwhm else 'N/A',
                      f'{peak.resolution:.4f}' if peak.resolution else 'N/A']
            for col, value in enumerate(values):
                self.table.setItem(k, col, QtWidgets.QTableWidgetItem(value))
            label = pg.TextItem(f'{peak.centroid:.2f}', color='#ffd584', anchor=(.5, 1.2))
            height = float(frame.spectrum[peak.maximum])
            label.setPos(peak.centroid, np.log10(max(1, height)) if self.log.isChecked() else height)
            self.plot.addItem(label)
            self.annotations.append(label)
        if self.autoscale.isChecked():
            self.plot.enableAutoRange()

    def save_csv(self, path):
        if self.current is None:
            raise ValueError('No validated frame to save')
        frame = self.current
        with open(path, 'w', newline='', encoding='utf-8') as output:
            writer = csv.writer(output)
            writer.writerow(['# MCA1', 'sequence', frame.sequence, 'duration_ms', frame.duration_ms, 'saturated', int(frame.saturated)])
            writer.writerow(['channel', 'counts'])
            writer.writerows(enumerate(map(int, frame.spectrum)))

    def save_dialog(self):
        if self.current is None:
            QtWidgets.QMessageBox.information(self, 'No spectrum', 'Receive a valid frame first.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save current spectrum', 'spectrum.csv', 'CSV (*.csv)')
        if path:
            try:
                self.save_csv(path)
            except OSError as error:
                QtWidgets.QMessageBox.warning(self, 'Save failed', str(error))

    def closeEvent(self, event):
        self.stop()
        self.timer.stop()
        super().closeEvent(event)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    viewer = Viewer(args.simulate, args.host, args.port)
    viewer.show()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
