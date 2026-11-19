#if defined(__GNUC__)
#pragma GCC optimize ("O2")
#endif
#include "mca_sim.h"
#include "mca_profile.h"
#include <math.h>

mca_sim_config_t mca_sim_default_config(void)
{
    mca_sim_config_t c = {0x7432048U, 8.0, 0.002, 1.0,
        {{400.0, 10.0, 800.0}, {1000.0, 16.0, 1200.0}, {1600.0, 22.0, 650.0}}};
    return c;
}
static uint32_t mix(uint32_t v)
{
    v ^= v >> 16; v *= UINT32_C(0x7feb352d); v ^= v >> 15;
    v *= UINT32_C(0x846ca68b); return v ^ (v >> 16);
}
/* Single-owner simulation cache. Exact original double formulas; invalidate
 * on every field change, never compare struct padding with memcmp. */
static double cached_rate[MCA_CHANNELS], cached_noise[MCA_CHANNELS];
static mca_sim_config_t cached_config;
static bool cache_valid;
static bool same_config(const mca_sim_config_t *c)
{
    if (!cache_valid || c->seed != cached_config.seed || c->background_cps != cached_config.background_cps ||
        c->background_slope_cps != cached_config.background_slope_cps || c->noise_scale != cached_config.noise_scale) { return false; }
    for (size_t k=0; k<MCA_SIM_PEAKS; ++k) {
        if (c->peaks[k].channel != cached_config.peaks[k].channel || c->peaks[k].sigma_channels != cached_config.peaks[k].sigma_channels ||
            c->peaks[k].height_cps != cached_config.peaks[k].height_cps) { return false; }
    }
    return true;
}
mca_status_t mca_simulate(mca_histogram_t *h, const mca_sim_config_t *cfg, uint64_t duration_ms)
{
    if (h == NULL || cfg == NULL || duration_ms == 0U ||
        !isfinite(cfg->background_cps) || cfg->background_cps < 0.0 || cfg->background_cps > 1e12 ||
        !isfinite(cfg->background_slope_cps) || fabs(cfg->background_slope_cps) > 1e9 ||
        cfg->background_cps + cfg->background_slope_cps*(MCA_CHANNELS-1U) < 0.0 ||
        !isfinite(cfg->noise_scale) || cfg->noise_scale < 0.0 || cfg->noise_scale > 10.0) { return MCA_INVALID; }
    for (size_t k = 0; k < MCA_SIM_PEAKS; ++k) {
        const mca_sim_peak_t *p = &cfg->peaks[k];
        if (!isfinite(p->channel) || p->channel < 0.0 || p->channel > MCA_CHANNELS-1U ||
            !isfinite(p->sigma_channels) || p->sigma_channels < 0.25 || p->sigma_channels > MCA_CHANNELS ||
            !isfinite(p->height_cps) || p->height_cps < 0.0 || p->height_cps > 1e12) { return MCA_INVALID; }
    }
    g_mca_timing.cache_build = 0U;
    if (!same_config(cfg)) {
        uint32_t start = mca_cycles();
        for (size_t i = 0; i < MCA_CHANNELS; ++i) {
        if ((i & 63U)==0U) { mca_service(); }
            double rate = cfg->background_cps + cfg->background_slope_cps*(double)i;
            for (size_t k = 0; k < MCA_SIM_PEAKS; ++k) {
                double z = ((double)i-cfg->peaks[k].channel)/cfg->peaks[k].sigma_channels;
                rate += cfg->peaks[k].height_cps*exp(-0.5*z*z);
            }
            cached_rate[i] = rate;
            double u = (double)mix(cfg->seed ^ (uint32_t)i)/4294967295.0;
            cached_noise[i] = (2.0*u-1.0)*1.7320508075688772*cfg->noise_scale;
        }
        cached_config = *cfg; cache_valid = true;
        g_mca_timing.cache_build = mca_cycles() - start;
    }
    h->saturated = false; h->duration_ms = duration_ms;
    double seconds = (double)duration_ms/1000.0;
    for (size_t i = 0; i < MCA_CHANNELS; ++i) {
        if ((i & 63U)==0U) { mca_service(); }
        double rate = cached_rate[i];
        double mean = rate*seconds;
        /* Bounded deterministic noise with variance approximately mean.
         * This is a repeatable test model, not an exact Poisson sampler. */
        double count = fmax(0.0, mean + cached_noise[i]*sqrt(mean));
        if (count >= (double)UINT32_MAX-0.5) { h->spectrum[i] = UINT32_MAX; h->saturated = true; }
        else { h->spectrum[i] = (uint32_t)(count+0.5); }
    }
    return h->saturated ? MCA_SATURATED : MCA_OK;
}
