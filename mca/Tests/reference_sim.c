/* Frozen phase-1 simulation oracle; do not optimize its formula. */
#include "mca_sim.h"
#include <math.h>

mca_sim_config_t reference_sim_default_config(void)
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
mca_status_t reference_simulate(mca_histogram_t *h, const mca_sim_config_t *cfg, uint64_t duration_ms)
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
    h->saturated = false; h->duration_ms = duration_ms;
    double seconds = (double)duration_ms/1000.0;
    for (size_t i = 0; i < MCA_CHANNELS; ++i) {
        double rate = cfg->background_cps + cfg->background_slope_cps*(double)i;
        for (size_t k = 0; k < MCA_SIM_PEAKS; ++k) {
            double z = ((double)i-cfg->peaks[k].channel)/cfg->peaks[k].sigma_channels;
            rate += cfg->peaks[k].height_cps*exp(-0.5*z*z);
        }
        double mean = rate*seconds;
        /* Bounded deterministic noise with variance approximately mean.
         * This is a repeatable test model, not an exact Poisson sampler. */
        double u = (double)mix(cfg->seed ^ (uint32_t)i)/4294967295.0;
        double count = fmax(0.0, mean + (2.0*u-1.0)*1.7320508075688772*cfg->noise_scale*sqrt(mean));
        if (count >= (double)UINT32_MAX-0.5) { h->spectrum[i] = UINT32_MAX; h->saturated = true; }
        else { h->spectrum[i] = (uint32_t)(count+0.5); }
    }
    return h->saturated ? MCA_SATURATED : MCA_OK;
}
