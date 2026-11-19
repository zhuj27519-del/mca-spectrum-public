#ifndef MCA_STREAM_H
#define MCA_STREAM_H
#include "mca_frame.h"
/* Single-owner NO_SYS transport. Writer returns copied bytes, 0 for backpressure,
 * -1 for disconnect. Never retain a pointer to the caller's frame. */
typedef int (*mca_stream_write_fn)(void *, const uint8_t *, size_t);
typedef struct {
    uint8_t frame[MCA_FRAME_SIZE];
    size_t offset;
    uint32_t sequence, last_offer, frames, bytes, blocked, errors, skipped;
    bool connected, active;
} mca_stream_t;
void mca_stream_init(mca_stream_t *s);
void mca_stream_connection(mca_stream_t *s, bool connected, uint32_t now);
bool mca_stream_offer(mca_stream_t *s, const mca_histogram_t *h,
                      const volatile uint32_t *generation, uint32_t now);
void mca_stream_pump(mca_stream_t *s, mca_stream_write_fn writer, void *context);
#endif
