"""MCA1 v1: exact 8224-byte little-endian wire format, no TCP message assumptions."""
from dataclasses import dataclass
import struct
import zlib
import numpy as np

CHANNELS, HEADER_SIZE, FRAME_SIZE = 2048, 32, 8224
HEADER = struct.Struct('<4sHHIQHHII')

@dataclass(frozen=True)
class Frame:
    sequence: int
    duration_ms: int
    spectrum: np.ndarray
    saturated: bool = False

    @property
    def total(self):
        return int(self.spectrum.sum(dtype=np.uint64))

    @property
    def cps(self):
        return self.total * 1000.0 / self.duration_ms

def encode(frame):
    bins = np.asarray(frame.spectrum)
    if bins.shape != (CHANNELS,) or not 0 < frame.duration_ms < 2**64:
        raise ValueError('Invalid channels or duration')
    if not np.issubdtype(bins.dtype, np.integer) or np.any(bins < 0) or np.any(bins > 0xffffffff):
        raise ValueError('Spectrum must contain uint32 counts')
    header = HEADER.pack(b'MCA1', 1, 32, frame.sequence & 0xffffffff, frame.duration_ms,
                         2048, int(frame.saturated), 8192, 0)
    payload = bins.astype('<u4').tobytes()
    crc = zlib.crc32(payload, zlib.crc32(header[:28]))
    return header[:28] + struct.pack('<I', crc) + payload

def valid_header(header):
    magic, version, size, seq, ms, channels, flags, length, crc = HEADER.unpack(header)
    return magic == b'MCA1' and version == 1 and size == 32 and ms > 0 and channels == 2048 and flags <= 1 and length == 8192

def decode(data):
    if len(data) != FRAME_SIZE or not valid_header(data[:32]):
        raise ValueError('Invalid MCA1 header/length')
    _, _, _, seq, ms, _, flags, _, crc = HEADER.unpack(data[:32])
    if zlib.crc32(data[32:], zlib.crc32(data[:28])) != crc:
        raise ValueError('CRC mismatch')
    bins = np.frombuffer(data, dtype='<u4', count=2048, offset=32).copy()
    bins.flags.writeable = False
    return Frame(seq, ms, bins, bool(flags))

class StreamParser:
    def __init__(self):
        self.buffer = bytearray()
        self.frames = self.crc_errors = self.header_errors = self.discarded_bytes = 0

    def feed(self, data):
        """Yield validated frames; bounded buffer even for arbitrarily large input."""
        for offset in range(0, len(data), FRAME_SIZE):
            self.buffer.extend(data[offset:offset + FRAME_SIZE])
            while len(self.buffer) >= 4:
                index = self.buffer.find(b'MCA1')
                if index < 0:
                    count = len(self.buffer) - 3
                    self.discarded_bytes += count
                    del self.buffer[:count]
                    break
                if index:
                    self.discarded_bytes += index
                    del self.buffer[:index]
                if len(self.buffer) < 32:
                    break
                if not valid_header(self.buffer[:32]):
                    self.header_errors += 1
                    del self.buffer[0]
                    continue
                if len(self.buffer) < FRAME_SIZE:
                    break
                try:
                    frame = decode(bytes(self.buffer[:FRAME_SIZE]))
                except ValueError:
                    self.crc_errors += 1
                    del self.buffer[0]
                    continue
                del self.buffer[:FRAME_SIZE]
                self.frames += 1
                yield frame
