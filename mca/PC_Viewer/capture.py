"""Bounded TCP/CRC diagnostic: python capture.py --host 127.0.0.1."""
import argparse
import json
import socket
import time
from dataclasses import asdict
from protocol import StreamParser
from analysis import analyze


def capture(host, port=5000, frames=100, timeout=20):
    if frames < 1 or timeout <= 0:
        raise ValueError('frames and timeout must be positive')
    parser = StreamParser()
    count = gaps = 0
    previous = last = None
    start = time.monotonic()
    error = None
    try:
        with socket.create_connection((host, port), timeout=min(timeout, 3)) as connection:
            while count < frames:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('Capture deadline exceeded')
                connection.settimeout(remaining)
                data = connection.recv(65536)
                if not data:
                    raise ConnectionError('Peer closed before requested frame count')
                for frame in parser.feed(data):
                    if previous is not None and frame.sequence != (previous + 1) & 0xffffffff:
                        gaps += 1
                    previous, last = frame.sequence, frame
                    count += 1
                    if count >= frames:
                        break
    except (OSError, TimeoutError, ConnectionError) as exc:
        error = str(exc)
    result = {'host': host, 'port': port, 'seconds': time.monotonic()-start,
              'frames': count, 'frame_bytes': 8224, 'channels': 2048,
              'crc_errors': parser.crc_errors, 'header_errors': parser.header_errors,
              'sequence_discontinuities': gaps, 'error': error,
              'pass': count >= frames and not error and parser.crc_errors == 0 and parser.header_errors == 0 and gaps == 0}
    if last is not None:
        peaks, _ = analyze(last.spectrum)
        result.update(sequence=last.sequence, duration_ms=last.duration_ms,
                      total=last.total, cps=last.cps, saturated=last.saturated,
                      peaks=[asdict(peak) for peak in peaks])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--output', default='capture-result.json')
    args = parser.parse_args()
    result = capture(args.host, args.port, args.frames, args.timeout)
    with open(args.output, 'w', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
