# MCA spectrum frame v1

Purpose: an identical transport-independent histogram snapshot can be received from the future FPGA or sent by STM32 to a PC. Phase 3 adds Ethernet TCP transport; no SPI/FMC FPGA driver is enabled. FPGA performs high-speed acquisition and histogram accumulation; STM32 receives completed snapshots and analyzes them. Do not attempt to stream 125 MSPS ADC samples into this API.

All integer fields are unsigned and LITTLE ENDIAN. Never transmit a C struct directly. Total frame size is exactly 8224 bytes. Channel index is 0..2047. A frame replaces the previous histogram; it is not an increment to it.

| Offset | Bytes | Field | Required value / meaning |
|---|---:|---|---|
| 0 | 4 | Magic | ASCII MCA1 (4d 43 41 31) |
| 4 | 2 | Version | 1 |
| 6 | 2 | Header length | 32 |
| 8 | 4 | Sequence | Increment modulo 2^32 per published frame |
| 12 | 8 | Acquisition duration | Milliseconds, nonzero; exposure associated with this snapshot |
| 20 | 2 | Channel count | 2048 |
| 22 | 2 | Flags | Bit 0: histogram has saturated; all other bits zero |
| 24 | 4 | Data length | 8192, excluding header |
| 28 | 4 | CRC-32 | CRC described below |
| 32 | 8192 | Spectrum | 2048 little-endian uint32 counts in increasing channel order |

CRC-32/ISO-HDLC: reflected polynomial 0xedb88320, initial value 0xffffffff, final XOR 0xffffffff. Calculate over header bytes 0..27 followed immediately by payload bytes 32..8223. Exclude the CRC field itself. Compatible with Python zlib.crc32(header[:28] + payload).

`mca_frame_encode` and `mca_frame_decode` are declared in Core/Inc/mca_frame.h. Encode requires capacity >=8224. Decode requires exactly 8224 bytes; rejects unknown flags/version, invalid dimensions/duration and CRC failure. On rejection it leaves the destination histogram and sequence unchanged. Source frame and destination histogram must not overlap.

Integration sequence:
1. Transport collects an entire frame into a separate receive buffer with framing/timeouts and length bounds. TCP is a byte stream, so multiple reads may be needed. This frame exceeds a normal Ethernet MTU; transport must segment/reassemble it. No assumption of a single UDP datagram or Ethernet frame.
2. Decode into an inactive histogram buffer. Sequence replay/stale-frame rejection and acquisition/session reset policy belong to the transport/application. Sequence wraps at UINT32_MAX; do not compare sequence numbers using plain signed greater-than across wrap.
3. Publish ownership of the completed snapshot to the main loop. Do not DMA or accumulate into a histogram while mca_analyze reads it. For future H7 DMA integration, choose DMA-accessible memory and implement cache coherency as required; Phase 3 uses D2 non-cacheable Ethernet DMA buffers; FPGA DMA remains future work.
4. Call mca_analyze using duration from the snapshot, not the packet arrival interval. CPS is total recorded counts / duration. Define whether the FPGA reports live time or real time before deployment; no dead-time correction is currently performed.
5. Encode the snapshot for the PC. This v1 frame transports bins and exposure; the PC may analyze them. Peak-result telemetry needs a separately versioned message rather than silently extending this fixed frame.

Saturating counters never wrap. If flag bit 0 is set, total counts/CPS are lower bounds and peak shape may be distorted. The FPGA must supply already mapped 2048-bin pulse-height data; the 12-bit ADC-to-energy mapping is an FPGA responsibility and is not assumed here.
