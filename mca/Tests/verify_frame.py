"""Independent wire-format oracle: Python standard library only."""
import pathlib
import struct
import zlib

frame = pathlib.Path('frame.bin').read_bytes()
assert len(frame) == 8224
magic, version, header, sequence, duration, channels, flags, length, crc = struct.unpack('<4sHHIQHHII', frame[:32])
assert (magic, version, header, sequence) == (b'MCA1', 1, 32, 0x12345678)
assert (duration, channels, flags, length) == (0x0102030405060708, 2048, 0, 8192)
assert crc == zlib.crc32(frame[:28] + frame[32:])
bins = struct.unpack('<2048I', frame[32:])
assert bins[0] == 0x12345678 and bins[-1] == 42 and sum(bins[1:-1]) == 0
print('PASS independent Python struct / zlib CRC-32 wire-format oracle')
