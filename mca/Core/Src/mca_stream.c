#include "mca_stream.h"
#include <string.h>
void mca_stream_init(mca_stream_t *s) { memset(s, 0, sizeof(*s)); }
void mca_stream_connection(mca_stream_t *s, bool connected, uint32_t now)
{
    s->connected=connected; s->active=false; s->offset=0;
    s->last_offer=now-100U;
}
bool mca_stream_offer(mca_stream_t *s, const mca_histogram_t *h,
                      const volatile uint32_t *generation, uint32_t now)
{
    if (!s || !h || !generation || !s->connected || (uint32_t)(now-s->last_offer)<100U) { return false; }
    if (s->active) { s->skipped++; return false; }
    uint32_t before=*generation;
    if ((before&1U) || !before) { return false; }
    /* Producer and caller run in the same thread. No ISR updates histogram.
     * Network service yields during CRC may close the connection. Recheck it. */
    if (mca_frame_encode(h,s->sequence,s->frame,sizeof(s->frame))!=MCA_OK) { s->errors++; return false; }
    if (*generation!=before || !s->connected) { s->skipped++; return false; }
    s->sequence++; s->offset=0; s->active=true; s->last_offer=now;
    return true;
}
void mca_stream_pump(mca_stream_t *s, mca_stream_write_fn writer, void *context)
{
    if (!s || !writer || !s->active || !s->connected) { return; }
    /* Bound work per service pass; remainder is retained until copied. */
    for (unsigned budget=0; budget<2U && s->active; ++budget) {
        size_t n=MCA_FRAME_SIZE-s->offset; if(n>1460U) { n=1460U; }
        int accepted=writer(context,s->frame+s->offset,n);
        if(accepted<0 || (size_t)accepted>n) {
            s->errors++; mca_stream_connection(s,false,s->last_offer); return;
        }
        if(accepted==0) { s->blocked++; return; }
        s->offset+=(size_t)accepted; s->bytes+=(uint32_t)accepted;
        if(s->offset==MCA_FRAME_SIZE) { s->active=false; s->frames++; }
    }
}
