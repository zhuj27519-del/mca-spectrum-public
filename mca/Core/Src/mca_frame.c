#include "mca_frame.h"
#include "mca_profile.h"
#include <string.h>
static void put(uint8_t *p, uint64_t v, size_t n)
{ for (size_t i = 0; i < n; ++i) { p[i] = (uint8_t)(v >> (8U*i)); } }
static uint64_t get(const uint8_t *p, size_t n)
{ uint64_t v = 0; for (size_t i = 0; i < n; ++i) { v |= (uint64_t)p[i] << (8U*i); } return v; }
static uint32_t crc(const uint8_t *p)
{
    uint32_t c = UINT32_MAX;
    for (size_t i = 0; i < MCA_FRAME_SIZE; ++i) {
        if (i >= 28U && i < 32U) { continue; }
        if ((i & 255U)==0U) { mca_service(); }
        c ^= p[i];
        for (unsigned b = 0; b < 8U; ++b) { c = (c >> 1) ^ ((c & 1U) ? UINT32_C(0xedb88320) : 0U); }
    }
    return c ^ UINT32_MAX;
}
mca_status_t mca_frame_encode(const mca_histogram_t *h, uint32_t sequence, uint8_t *frame, size_t capacity)
{
    if (h == NULL || frame == NULL || capacity < MCA_FRAME_SIZE || h->duration_ms == 0U) { return MCA_INVALID; }
    memcpy(frame, "MCA1", 4U); put(frame+4, 1U, 2U); put(frame+6, MCA_FRAME_HEADER, 2U);
    put(frame+8, sequence, 4U); put(frame+12, h->duration_ms, 8U);
    put(frame+20, MCA_CHANNELS, 2U); put(frame+22, h->saturated ? 1U : 0U, 2U);
    put(frame+24, MCA_CHANNELS*4U, 4U);
    for (size_t i = 0; i < MCA_CHANNELS; ++i) { put(frame+MCA_FRAME_HEADER+4U*i, h->spectrum[i], 4U); }
    put(frame+28, crc(frame), 4U); return MCA_OK;
}
mca_status_t mca_frame_decode(const uint8_t *frame, size_t length, mca_histogram_t *h, uint32_t *sequence)
{
    if (frame == NULL || h == NULL || sequence == NULL) { return MCA_INVALID; }
    if (length != MCA_FRAME_SIZE || memcmp(frame, "MCA1", 4U) != 0 || get(frame+4, 2U) != 1U ||
        get(frame+6, 2U) != MCA_FRAME_HEADER || get(frame+12, 8U) == 0U ||
        get(frame+20, 2U) != MCA_CHANNELS || get(frame+22, 2U) > 1U ||
        get(frame+24, 4U) != MCA_CHANNELS*4U || get(frame+28, 4U) != crc(frame)) { return MCA_BAD_FRAME; }
    for (size_t i = 0; i < MCA_CHANNELS; ++i) { h->spectrum[i] = (uint32_t)get(frame+MCA_FRAME_HEADER+4U*i, 4U); }
    h->duration_ms = get(frame+12, 8U); h->saturated = get(frame+22, 2U) != 0U;
    *sequence = (uint32_t)get(frame+8, 4U); return MCA_OK;
}
