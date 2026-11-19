#ifndef MCA_SIM_H
#define MCA_SIM_H
#include "mca.h"
#define MCA_SIM_PEAKS 3U
typedef struct { double channel, sigma_channels, height_cps; } mca_sim_peak_t;
typedef struct {
    uint32_t seed;
    double background_cps, background_slope_cps, noise_scale;
    mca_sim_peak_t peaks[MCA_SIM_PEAKS];
} mca_sim_config_t;
mca_sim_config_t mca_sim_default_config(void);
/* Replace histogram with a deterministic cumulative snapshot at this exposure.
 * Rates are counts / channel / second, sigma is NOT FWHM. Same inputs repeat.
 * noise_scale=0 produces an analytic rounded spectrum for quantitative tests. */
mca_status_t mca_simulate(mca_histogram_t *h, const mca_sim_config_t *cfg, uint64_t duration_ms);
#endif
