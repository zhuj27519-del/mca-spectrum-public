#ifndef MCA_H
#define MCA_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define MCA_CHANNELS 2048U
#define MCA_MAX_PEAKS 8U
typedef enum { MCA_OK = 0, MCA_INVALID, MCA_SATURATED, MCA_BAD_FRAME } mca_status_t;
typedef struct {
    uint32_t spectrum[MCA_CHANNELS];
    uint64_t duration_ms;
    bool saturated;
} mca_histogram_t;
typedef struct { bool enabled; double kev_per_channel; double offset_kev; } mca_calibration_t;
typedef struct {
    uint16_t background_radius, roi_radius, min_separation;
    float minimum_height, minimum_snr;
    float minimum_prominence, prominence_fraction;
} mca_analysis_config_t;
typedef struct {
    uint16_t maximum_channel;
    double centroid_channel, net_counts, fwhm_channels;
    double channel_resolution_percent;
    double energy_kev, fwhm_kev, energy_resolution_percent;
    bool fwhm_valid, channel_resolution_valid, energy_valid, energy_resolution_valid;
    bool roi_clipped;
} mca_peak_t;
typedef struct {
    uint64_t total_counts, duration_ms;
    double cps;
    bool cps_valid, saturated, peaks_truncated;
    uint16_t peak_count;
    mca_peak_t peaks[MCA_MAX_PEAKS];
} mca_result_t;
/* Caller-owned scratch: never allocate these arrays on the MCU stack. */
typedef struct {
    float smooth[MCA_CHANNELS], background[MCA_CHANNELS], net[MCA_CHANNELS];
    float scratch[MCA_CHANNELS];
    uint16_t deque[MCA_CHANNELS];
} mca_workspace_t;

void mca_init(mca_histogram_t *h);
void mca_clear(mca_histogram_t *h);
mca_status_t mca_add(mca_histogram_t *h, size_t channel, uint32_t counts);
mca_status_t mca_read(const mca_histogram_t *h, size_t channel, uint32_t *counts);
uint64_t mca_total(const mca_histogram_t *h);
mca_status_t mca_set_duration(mca_histogram_t *h, uint64_t milliseconds);
mca_status_t mca_load(mca_histogram_t *h, const uint32_t *bins, size_t channels, uint64_t milliseconds);
mca_status_t mca_set_calibration(mca_calibration_t *c, bool enabled, double slope, double offset);
mca_analysis_config_t mca_default_analysis_config(void);
mca_status_t mca_analyze(const mca_histogram_t *h, const mca_analysis_config_t *cfg,
                         const mca_calibration_t *cal, mca_workspace_t *w, mca_result_t *out);
#endif
