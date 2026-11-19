#if defined(__GNUC__)
/* Optimize only this portable numerical module; retain Debug symbols/no fast-math. */
#pragma GCC optimize ("O2")
#endif
#include "mca.h"
#include "mca_profile.h"
#include <math.h>
#include <string.h>
#include <float.h>

void mca_clear(mca_histogram_t *h) { if (h != NULL) { memset(h, 0, sizeof(*h)); } }
void mca_init(mca_histogram_t *h) { mca_clear(h); }
mca_status_t mca_add(mca_histogram_t *h, size_t channel, uint32_t counts)
{
    if (h == NULL || channel >= MCA_CHANNELS) { return MCA_INVALID; }
    if (counts > UINT32_MAX - h->spectrum[channel]) {
        h->spectrum[channel] = UINT32_MAX; h->saturated = true; return MCA_SATURATED;
    }
    h->spectrum[channel] += counts;
    return MCA_OK;
}
mca_status_t mca_read(const mca_histogram_t *h, size_t channel, uint32_t *counts)
{
    if (h == NULL || counts == NULL || channel >= MCA_CHANNELS) { return MCA_INVALID; }
    *counts = h->spectrum[channel]; return MCA_OK;
}
uint64_t mca_total(const mca_histogram_t *h)
{
    uint64_t sum = 0;
    if (h != NULL) { for (size_t i = 0; i < MCA_CHANNELS; ++i) {
        if ((i & 63U)==0U) { mca_service(); } sum += h->spectrum[i]; } }
    return sum;
}
mca_status_t mca_set_duration(mca_histogram_t *h, uint64_t milliseconds)
{
    if (h == NULL || milliseconds == 0U) { return MCA_INVALID; }
    h->duration_ms = milliseconds; return MCA_OK;
}
mca_status_t mca_load(mca_histogram_t *h, const uint32_t *bins, size_t channels, uint64_t milliseconds)
{
    if (h == NULL || bins == NULL || channels != MCA_CHANNELS || milliseconds == 0U) { return MCA_INVALID; }
    memmove(h->spectrum, bins, sizeof(h->spectrum)); h->duration_ms = milliseconds;
    h->saturated = false;
    for (size_t i = 0; i < MCA_CHANNELS; ++i) {
        if ((i & 63U)==0U) { mca_service(); } if (h->spectrum[i] == UINT32_MAX) { h->saturated = true; } }
    return MCA_OK;
}
mca_status_t mca_set_calibration(mca_calibration_t *c, bool enabled, double slope, double offset)
{
    if (c == NULL || (enabled && (!isfinite(slope) || !isfinite(offset) || slope <= 0.0 ||
        !isfinite(slope * (double)MCA_CHANNELS + fabs(offset))))) { return MCA_INVALID; }
    c->enabled = enabled; c->kev_per_channel = enabled ? slope : 0.0;
    c->offset_kev = enabled ? offset : 0.0; return MCA_OK;
}
mca_analysis_config_t mca_default_analysis_config(void)
{
    mca_analysis_config_t c = {96U, 80U, 60U, 20.0f, 5.0f, 15.0f, 0.05f}; return c;
}
static size_t lower(size_t i, size_t r) { return i > r ? i - r : 0U; }
static size_t upper(size_t i, size_t r) { return i + r < MCA_CHANNELS ? i + r : MCA_CHANNELS - 1U; }

/* Monotonic deque: identical window extrema in O(channels), rather than
 * O(channels * radius). Each input index is inserted and removed at most once. */
static void sliding_extreme(const float *input, float *output, size_t radius,
                            uint16_t *q, bool maximum)
{
    size_t head = 0U, tail = 0U, next = 0U;
    for (size_t i = 0U; i < MCA_CHANNELS; ++i) {
        if ((i & 63U)==0U) { mca_service(); }
        size_t hi = upper(i, radius), lo = lower(i, radius);
        while (next <= hi) {
            while (tail > head && (maximum ? input[q[tail-1U]] <= input[next] :
                                            input[q[tail-1U]] >= input[next])) { --tail; }
            q[tail++] = (uint16_t)next++;
        }
        while (head < tail && q[head] < lo) { ++head; }
        output[i] = input[q[head]];
    }
}
/* Five-bin smoothing followed by morphological opening (erosion then dilation).
 * Background window must exceed the width of resolved peaks. No heap, no HAL. */
static void estimate_background(const mca_histogram_t *h, size_t radius, mca_workspace_t *w)
{
    for (size_t i = 0; i < MCA_CHANNELS; ++i) {
        if ((i & 63U)==0U) { mca_service(); }
        size_t lo = lower(i, 2U), hi = upper(i, 2U); double sum = 0.0;
        for (size_t j = lo; j <= hi; ++j) { sum += h->spectrum[j]; }
        w->smooth[i] = (float)(sum / (double)(hi - lo + 1U));
    }
    sliding_extreme(w->smooth, w->scratch, radius, w->deque, false);
    sliding_extreme(w->scratch, w->background, radius, w->deque, true);
    for (size_t i = 0; i < MCA_CHANNELS; ++i) {
        if ((i & 63U)==0U) { mca_service(); }
        w->net[i] = fmaxf(0.0f, w->smooth[i] - w->background[i]);
    }
}
/* Prominence against the higher of the two surrounding valley floors. Stop at
 * a higher peak, so a small ripple on a large peak's tail cannot reuse the
 * distant background as its baseline. Edge peaks use the available side. */
static float prominence(const float *net, size_t i)
{
    float v = net[i], left = v, right = v;
    for (size_t j = i; j > 0U;) {
        if ((j & 63U)==0U) { mca_service(); }
        --j; if (net[j] > v) { break; }
        if (net[j] < left) { left = net[j]; }
    }
    for (size_t j = i+1U; j < MCA_CHANNELS; ++j) {
        if ((j & 63U)==0U) { mca_service(); }
        if (net[j] > v) { break; }
        if (net[j] < right) { right = net[j]; }
    }
    if (i == 0U) { left = 0.0f; }
    if (i == MCA_CHANNELS-1U) { right = 0.0f; }
    return v - fmaxf(left, right);
}

mca_status_t mca_analyze(const mca_histogram_t *h, const mca_analysis_config_t *cfg,
                         const mca_calibration_t *cal, mca_workspace_t *w, mca_result_t *out)
{
    if (h == NULL || cfg == NULL || cal == NULL || w == NULL || out == NULL) { return MCA_INVALID; }
    uint32_t analysis_start = mca_cycles();
    g_mca_timing.centroid = 0; g_mca_timing.fwhm = 0;
    memset(out, 0, sizeof(*out));
    if (cfg->background_radius < 3U || cfg->background_radius >= MCA_CHANNELS ||
        cfg->roi_radius < 2U || cfg->roi_radius >= MCA_CHANNELS ||
        cfg->min_separation == 0U || cfg->min_separation >= MCA_CHANNELS ||
        !isfinite(cfg->minimum_height) || cfg->minimum_height <= 0.0f ||
        !isfinite(cfg->minimum_snr) || cfg->minimum_snr < 0.0f ||
        !isfinite(cfg->minimum_prominence) || cfg->minimum_prominence < 0.0f ||
        !isfinite(cfg->prominence_fraction) || cfg->prominence_fraction < 0.0f || cfg->prominence_fraction > 1.0f) { return MCA_INVALID; }
    mca_calibration_t checked;
    if (mca_set_calibration(&checked, cal->enabled, cal->kev_per_channel, cal->offset_kev) != MCA_OK) { return MCA_INVALID; }
    out->total_counts = mca_total(h); out->duration_ms = h->duration_ms; out->saturated = h->saturated;
    out->cps_valid = h->duration_ms != 0U;
    if (out->cps_valid) { out->cps = (double)out->total_counts * 1000.0 / (double)h->duration_ms; }
    uint32_t stage_start = mca_cycles();
    estimate_background(h, cfg->background_radius, w);
    g_mca_timing.background = mca_cycles() - stage_start;
    stage_start = mca_cycles();
    /* Estimate excess noise from adjacent raw bins in background-only regions.
     * Poisson differences have variance count[i]+count[i-1]. This also handles
     * overdispersed simulated/real backgrounds without fixing the peak count. */
    double difference_power = 0.0, poisson_power = 0.0;
    for (size_t i=1U; i<MCA_CHANNELS; ++i) {
        if (w->net[i] <= cfg->minimum_snr*sqrtf(w->background[i]+1.0f) &&
            w->net[i-1U] <= cfg->minimum_snr*sqrtf(w->background[i-1U]+1.0f)) {
            double d = (double)h->spectrum[i] - h->spectrum[i-1U] -
                       ((double)w->background[i]-w->background[i-1U]);
            difference_power += d*d;
            poisson_power += (double)h->spectrum[i]+h->spectrum[i-1U];
        }
    }
    float noise_factor = poisson_power > 0.0 ? (float)sqrt(difference_power/poisson_power) : 1.0f;
    noise_factor = fmaxf(1.0f, noise_factor);
    /* Select strongest separated maxima; include edge maxima, flag their clipped ROIs. */
    for (size_t pass = 0; pass <= MCA_MAX_PEAKS; ++pass) {
        size_t best = MCA_CHANNELS; float height = 0.0f;
        for (size_t i = 0; i < MCA_CHANNELS; ++i) {
        if ((i & 63U)==0U) { mca_service(); }
            float v = w->net[i];
            if (v < cfg->minimum_height || v < cfg->minimum_snr * sqrtf(w->background[i] + 1.0f) ||
                (i > 0U && v <= w->net[i-1U]) || (i+1U < MCA_CHANNELS && v < w->net[i+1U])) { continue; }
            float prom = prominence(w->net, i);
            if (prom < cfg->minimum_prominence || prom < cfg->prominence_fraction*v ||
                prom < noise_factor*cfg->minimum_snr*sqrtf(w->smooth[i]+1.0f)) { continue; }
            bool near = false;
            for (size_t k = 0; k < out->peak_count; ++k) {
                size_t p = out->peaks[k].maximum_channel;
                size_t distance = i > p ? i - p : p - i;
                if (distance < cfg->min_separation) { near = true; }
            }
            if (!near && v > height) { best = i; height = v; }
        }
        if (best == MCA_CHANNELS) { break; }
        if (pass == MCA_MAX_PEAKS) { out->peaks_truncated = true; break; }
        out->peaks[out->peak_count++].maximum_channel = (uint16_t)best;
    }
    g_mca_timing.detection = mca_cycles() - stage_start;
    /* Present peaks in ascending channel order. */
    for (size_t a = 1; a < out->peak_count; ++a) {
        mca_peak_t p = out->peaks[a]; size_t b = a;
        while (b > 0U && out->peaks[b-1U].maximum_channel > p.maximum_channel) {
            out->peaks[b] = out->peaks[b-1U]; --b;
        }
        out->peaks[b] = p;
    }
    for (size_t k = 0; k < out->peak_count; ++k) {
        stage_start = mca_cycles();
        mca_peak_t *p = &out->peaks[k]; size_t m = p->maximum_channel;
        size_t lo = lower(m, cfg->roi_radius), hi = upper(m, cfg->roi_radius);
        p->roi_clipped = m < cfg->roi_radius || m + cfg->roi_radius >= MCA_CHANNELS;
        if (k > 0U) { size_t mid = (m + out->peaks[k-1U].maximum_channel)/2U + 1U;
            if (mid > lo) { lo = mid; p->roi_clipped = true; } }
        if (k+1U < out->peak_count) { size_t mid = (m + out->peaks[k+1U].maximum_channel)/2U;
            if (mid < hi) { hi = mid; p->roi_clipped = true; } }
        double weighted = 0.0;
        for (size_t j = lo; j <= hi; ++j) {
            double net = fmax(0.0, (double)h->spectrum[j] - w->background[j]);
            p->net_counts += net; weighted += (double)j * net;
        }
        p->centroid_channel = p->net_counts > 0.0 ? weighted/p->net_counts : (double)m;
        g_mca_timing.centroid += mca_cycles() - stage_start;
        stage_start = mca_cycles();
        double half = w->net[m] * 0.5; size_t l = m, r = m;
        while (l > lo && w->net[l] > half) { --l; }
        while (r < hi && w->net[r] > half) { ++r; }
        if (l < m && r > m && w->net[l] <= half && w->net[r] <= half) {
            double left = (double)l + (half-w->net[l])/(w->net[l+1U]-w->net[l]);
            double right = (double)(r-1U) + (w->net[r-1U]-half)/(w->net[r-1U]-w->net[r]);
            p->fwhm_channels = right-left; p->fwhm_valid = p->fwhm_channels > 0.0;
        }
        g_mca_timing.fwhm += mca_cycles() - stage_start;
        if (p->fwhm_valid && p->centroid_channel > 0.0) {
            p->channel_resolution_percent = 100.0*p->fwhm_channels/p->centroid_channel;
            p->channel_resolution_valid = isfinite(p->channel_resolution_percent);
        }
        if (checked.enabled) {
            p->energy_kev = checked.kev_per_channel*p->centroid_channel + checked.offset_kev;
            p->energy_valid = isfinite(p->energy_kev) && p->energy_kev > 0.0;
            if (p->fwhm_valid) { p->fwhm_kev = checked.kev_per_channel*p->fwhm_channels; }
            if (p->energy_valid && p->fwhm_valid) {
                double resolution = (p->fwhm_kev / p->energy_kev)*100.0;
                p->energy_resolution_valid = isfinite(resolution);
                if (p->energy_resolution_valid) { p->energy_resolution_percent = resolution; }
            }
        }
    }
    g_mca_timing.other = mca_cycles() - analysis_start - g_mca_timing.background - g_mca_timing.detection - g_mca_timing.centroid - g_mca_timing.fwhm;
    return MCA_OK;
}
