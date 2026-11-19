#ifndef MCA_FRAME_H
#define MCA_FRAME_H
#include "mca.h"
#define MCA_FRAME_HEADER 32U
#define MCA_FRAME_SIZE (MCA_FRAME_HEADER + MCA_CHANNELS*4U)
/* Caller-owned buffers; frame and histogram must not overlap. Decode is
 * transactional: invalid frames leave histogram and sequence unchanged. */
mca_status_t mca_frame_encode(const mca_histogram_t *h, uint32_t sequence, uint8_t *frame, size_t capacity);
mca_status_t mca_frame_decode(const uint8_t *frame, size_t length, mca_histogram_t *h, uint32_t *sequence);
#endif
