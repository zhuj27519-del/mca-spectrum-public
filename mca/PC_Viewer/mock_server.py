"""Real loopback TCP peer for integration tests; never represents hardware."""
import argparse
import socket
import threading
import time
from protocol import encode
from simulation import simulate

class MockServer:
    def __init__(self, host='127.0.0.1', port=5000, interval=.1, disconnect_every=0, fragmented=True):
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.port = self.socket.getsockname()[1]
        self.socket.listen(2)
        self.socket.settimeout(.2)
        self.interval, self.disconnect_every, self.fragmented = interval, disconnect_every, fragmented
        self.stop_event = threading.Event()
        self.connections = self.frames = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def run(self):
        sequence = 0
        while not self.stop_event.is_set():
            try:
                client, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connections += 1
            with client:
                client.settimeout(.5)
                session_frames = 0
                while not self.stop_event.is_set():
                    sequence += 1
                    data = encode(simulate(1000+sequence*100, sequence))
                    try:
                        if self.fragmented:
                            for a, b in [(0, 3), (3, 31), (31, 1024), (1024, len(data))]:
                                client.sendall(data[a:b])
                        else:
                            client.sendall(data)
                    except OSError:
                        break
                    session_frames += 1
                    self.frames += 1
                    if self.disconnect_every and session_frames >= self.disconnect_every:
                        break
                    self.stop_event.wait(self.interval)

    def close(self):
        self.stop_event.set()
        self.socket.close()
        self.thread.join(timeout=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--disconnect-every', type=int, default=0)
    args = parser.parse_args()
    server = MockServer(port=args.port, disconnect_every=args.disconnect_every).start()
    print(f'LOCAL TEST SERVER 127.0.0.1:{server.port}; Ctrl+C to stop', flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.close()
