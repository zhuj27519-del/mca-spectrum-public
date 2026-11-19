from capture import capture
from mock_server import MockServer


def test_capture_fragmented_tcp():
    server = MockServer(port=0, interval=.01).start()
    try:
        result = capture('127.0.0.1', server.port, frames=5, timeout=3)
        assert result['pass'] and result['frames'] == 5
        assert len(result['peaks']) == 3
        assert result['cps'] > 0
    finally:
        server.close()


def test_capture_disconnect_is_not_pass():
    server = MockServer(port=0, interval=.01, disconnect_every=2).start()
    try:
        result = capture('127.0.0.1', server.port, frames=5, timeout=3)
        assert not result['pass'] and result['frames'] == 2 and result['error']
    finally:
        server.close()
