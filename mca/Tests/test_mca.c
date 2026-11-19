#include "mca.h"
#include "mca_sim.h"
#include "mca_frame.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static mca_histogram_t h, copy;
static mca_workspace_t w;
static mca_result_t r;
static uint8_t frame[MCA_FRAME_SIZE];
static void histogram_tests(void)
{
    uint32_t value = 123U;
    mca_init(&h); CHECK(mca_total(&h) == 0U);
    CHECK(mca_add(&h, 0U, 3U) == MCA_OK);
    CHECK(mca_add(&h, 2047U, 7U) == MCA_OK);
    CHECK(mca_total(&h) == 10U);
    CHECK(mca_read(&h, 2047U, &value) == MCA_OK && value == 7U);
    CHECK(mca_add(&h, 2048U, 10U) == MCA_INVALID);
    CHECK(mca_add(&h, SIZE_MAX, 1U) == MCA_INVALID);
    CHECK(mca_read(&h, 2048U, &value) == MCA_INVALID && value == 7U);
    CHECK(mca_add(NULL, 0U, 1U) == MCA_INVALID);
    CHECK(mca_read(&h, 0U, NULL) == MCA_INVALID);
    CHECK(mca_set_duration(&h, 0U) == MCA_INVALID);
    CHECK(mca_add(&h, 0U, UINT32_MAX) == MCA_SATURATED);
    CHECK(h.spectrum[0] == UINT32_MAX && h.saturated);
    for (size_t i = 0; i < MCA_CHANNELS; ++i) { h.spectrum[i] = UINT32_MAX; }
    CHECK(mca_total(&h) == (uint64_t)UINT32_MAX*MCA_CHANNELS);
    CHECK(mca_load(&copy, h.spectrum, MCA_CHANNELS, UINT64_MAX) == MCA_OK);
    CHECK(copy.saturated && copy.duration_ms == UINT64_MAX);
    CHECK(mca_load(&copy, h.spectrum, 0U, 1U) == MCA_INVALID);
    mca_clear(&h); CHECK(!h.saturated && h.duration_ms == 0U && mca_total(&h) == 0U);
    puts("PASS histogram / overflow / bounds");
}
static void analysis_tests(void)
{
    mca_analysis_config_t cfg = mca_default_analysis_config();
    mca_calibration_t cal = {false, 0.0, 0.0};
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK);
    CHECK(r.peak_count == 0U && !r.cps_valid && r.cps == 0.0);
    CHECK(mca_set_duration(&h, 2000U) == MCA_OK);
    CHECK(mca_add(&h, 100U, 200U) == MCA_OK);
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK);
    CHECK(r.total_counts == 200U && r.cps_valid && fabs(r.cps-100.0) < 1e-9);
    h.duration_ms = UINT64_MAX;
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK && isfinite(r.cps));
    cfg.roi_radius = 0U; CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_INVALID);
    cfg = mca_default_analysis_config(); cfg.minimum_snr = NAN;
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_INVALID);
    cfg = mca_default_analysis_config();
    CHECK(mca_set_calibration(&cal, true, -1.0, 0.0) == MCA_INVALID);
    CHECK(mca_set_calibration(&cal, true, DBL_MAX, 0.0) == MCA_INVALID);
    CHECK(mca_set_calibration(&cal, true, NAN, 0.0) == MCA_INVALID);
    CHECK(mca_set_calibration(&cal, false, 0.0, 0.0) == MCA_OK);
    mca_sim_config_t sim = mca_sim_default_config(); sim.noise_scale = 0.0;
    CHECK(mca_simulate(&h, &sim, 10000U) == MCA_OK);
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK && r.peak_count == 3U);
    for (size_t k = 0; k < 3U; ++k) {
        CHECK(fabs(r.peaks[k].centroid_channel-sim.peaks[k].channel) < 0.25);
        CHECK(r.peaks[k].fwhm_valid);
        /* Independent Gaussian oracle; allow five-bin smoothing broadening. */
        CHECK(fabs(r.peaks[k].fwhm_channels-2.354820045*sim.peaks[k].sigma_channels) < 0.5);
        CHECK(!r.peaks[k].energy_valid && r.peaks[k].energy_kev == 0.0);
        CHECK(r.peaks[k].net_counts > 0.0 && r.peaks[k].channel_resolution_valid);
        CHECK(fabs(w.background[(size_t)sim.peaks[k].channel] -
            10.0*(sim.background_cps+sim.background_slope_cps*sim.peaks[k].channel)) < 5.0);
    }
    CHECK(fabs(r.cps-(double)mca_total(&h)/10.0) < 1e-8);
    CHECK(mca_set_calibration(&cal, true, 0.5, 10.0) == MCA_OK);
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK);
    for (size_t k = 0; k < 3U; ++k) {
        CHECK(r.peaks[k].energy_valid && r.peaks[k].energy_resolution_valid);
        CHECK(fabs(r.peaks[k].energy_kev-(0.5*r.peaks[k].centroid_channel+10.0)) < 1e-8);
        CHECK(fabs(r.peaks[k].energy_resolution_percent-
            100.0*0.5*r.peaks[k].fwhm_channels/r.peaks[k].energy_kev) < 1e-8);
    }
    CHECK(mca_set_calibration(&cal, true, 0.5, -10000.0) == MCA_OK);
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK && !r.peaks[0].energy_valid);
    sim.peaks[0].channel = 0.0; sim.peaks[1].height_cps = 0.0;
    sim.peaks[2].channel = 2047.0;
    CHECK(mca_simulate(&h, &sim, 1000U) == MCA_OK);
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK && r.peak_count == 2U);
    CHECK(r.peaks[0].roi_clipped && !r.peaks[0].fwhm_valid);
    CHECK(r.peaks[1].roi_clipped && !r.peaks[1].fwhm_valid);
    mca_clear(&h);
    for (size_t i = 0; i < MCA_CHANNELS; ++i) { h.spectrum[i] = 100U; }
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK && r.peak_count == 0U);
    for (size_t i = 0; i < MCA_CHANNELS; ++i) { h.spectrum[i] = UINT32_MAX; }
    h.saturated = true; h.duration_ms = 1U;
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK && r.peak_count == 0U);
    CHECK(r.saturated && isfinite(r.cps) && r.total_counts == (uint64_t)UINT32_MAX*MCA_CHANNELS);
    mca_clear(&h); cfg.minimum_height = 1.0f; cfg.minimum_snr = 0.0f;
    for (size_t i = 100U; i < 1900U; i += 150U) { h.spectrum[i] = 1000U; }
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK);
    CHECK(r.peak_count == MCA_MAX_PEAKS && r.peaks_truncated);
    /* A narrow ROI must not fabricate a half-height crossing. */
    sim = mca_sim_default_config(); sim.noise_scale = 0.0;
    CHECK(mca_simulate(&h, &sim, 1000U) == MCA_OK);
    cfg = mca_default_analysis_config(); cfg.roi_radius = 2U;
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK && r.peak_count == 3U);
    CHECK(!r.peaks[0].fwhm_valid && !r.peaks[1].fwhm_valid && !r.peaks[2].fwhm_valid);
    puts("PASS CPS / background / centroid / FWHM / calibration / edge peaks / limits");
}
static void simulation_tests(void)
{
    mca_sim_config_t sim = mca_sim_default_config();
    mca_analysis_config_t cfg = mca_default_analysis_config(); mca_calibration_t cal = {false,0,0};
    CHECK(mca_simulate(&h, &sim, 1000U) == MCA_OK);
    CHECK(mca_simulate(&copy, &sim, 1000U) == MCA_OK);
    CHECK(memcmp(h.spectrum, copy.spectrum, sizeof(h.spectrum)) == 0);
    CHECK(mca_analyze(&h, &cfg, &cal, &w, &r) == MCA_OK && r.peak_count == 3U);
    for (size_t k = 0; k < 3U; ++k) {
        CHECK(fabs(r.peaks[k].centroid_channel-sim.peaks[k].channel) < 1.0);
        CHECK(fabs(r.peaks[k].fwhm_channels-2.354820045*sim.peaks[k].sigma_channels) < 2.0);
        printf("Sim peak %u: centroid=%.4f FWHM=%.4f channels\n", (unsigned)k,
               r.peaks[k].centroid_channel, r.peaks[k].fwhm_channels);
    }
    ++sim.seed; CHECK(mca_simulate(&copy, &sim, 1000U) == MCA_OK);
    CHECK(memcmp(h.spectrum, copy.spectrum, sizeof(h.spectrum)) != 0);
    copy = h; sim.peaks[0].sigma_channels = 0.0;
    CHECK(mca_simulate(&h, &sim, 1000U) == MCA_INVALID);
    CHECK(memcmp(h.spectrum, copy.spectrum, sizeof(h.spectrum)) == 0);
    sim = mca_sim_default_config();
    CHECK(mca_simulate(&h, &sim, 0U) == MCA_INVALID);
    CHECK(mca_simulate(&h, &sim, UINT64_MAX) == MCA_SATURATED && h.saturated);
    sim.background_slope_cps = -1.0;
    CHECK(mca_simulate(&h, &sim, 1000U) == MCA_INVALID);
    puts("PASS deterministic simulation / noise / invalid settings / saturation");
}
static void frame_tests(void)
{
    uint32_t seq = 0U;
    mca_clear(&h); h.spectrum[0] = 0x12345678U; h.spectrum[2047] = 42U; h.duration_ms = 0x0102030405060708ULL;
    CHECK(mca_frame_encode(&h, 0x12345678U, frame, sizeof(frame)) == MCA_OK);
    CHECK(frame[0]=='M' && frame[4]==1U && frame[6]==32U && frame[8]==0x78U);
    CHECK(frame[12]==8U && frame[19]==1U && frame[20]==0U && frame[21]==8U);
    CHECK(frame[32]==0x78U && frame[35]==0x12U);
    CHECK(mca_frame_decode(frame, sizeof(frame), &copy, &seq) == MCA_OK);
    CHECK(seq==0x12345678U && copy.duration_ms==h.duration_ms);
    CHECK(memcmp(h.spectrum,copy.spectrum,sizeof(h.spectrum))==0);
    CHECK(mca_frame_decode(frame, 1U, &copy, &seq)==MCA_BAD_FRAME);
    CHECK(mca_frame_decode(frame, sizeof(frame)-1U, &copy, &seq)==MCA_BAD_FRAME);
    CHECK(mca_frame_encode(&h, 0U, frame, sizeof(frame)-1U)==MCA_INVALID);
    const size_t positions[] = {0U,4U,6U,8U,12U,20U,22U,24U,28U,32U,8223U};
    for(size_t i=0; i<sizeof(positions)/sizeof(positions[0]); ++i) {
        frame[positions[i]] ^= 0xffU;
        CHECK(mca_frame_decode(frame,sizeof(frame),&copy,&seq)==MCA_BAD_FRAME);
        CHECK(seq==0x12345678U && memcmp(h.spectrum,copy.spectrum,sizeof(h.spectrum))==0);
        frame[positions[i]] ^= 0xffU;
    }
    FILE *f = fopen("frame.bin","wb"); CHECK(f != NULL);
    CHECK(fwrite(frame,1,sizeof(frame),f)==sizeof(frame)); CHECK(fclose(f)==0);
    puts("PASS frame roundtrip / endian / corruption / transactional rejection");
}
mca_status_t reference_simulate(mca_histogram_t *, const mca_sim_config_t *, uint64_t);
static void phase2_tests(void)
{
    mca_sim_config_t sim = mca_sim_default_config();
    mca_analysis_config_t cfg = mca_default_analysis_config();
    mca_calibration_t cal = {false,0,0};
    const uint64_t durations[] = {500U,1000U,2078U,10000U,60000U,3600000U};
    const double noises[] = {0.0,0.5,1.0,2.0};
    unsigned scenarios = 0U;
    for (size_t a=0; a<6; ++a) for (size_t b=0; b<4; ++b) for (uint32_t seed=1; seed<=8; ++seed) {
        sim.seed=seed; sim.noise_scale=noises[b];
        CHECK(mca_simulate(&h,&sim,durations[a])==MCA_OK);
        CHECK(reference_simulate(&copy,&sim,durations[a])==MCA_OK);
        CHECK(memcmp(h.spectrum,copy.spectrum,sizeof(h.spectrum))==0);
        /* A repeated call must reuse the cache without altering any bin. */
        CHECK(mca_simulate(&h,&sim,durations[a])==MCA_OK);
        CHECK(memcmp(h.spectrum,copy.spectrum,sizeof(h.spectrum))==0);
        CHECK(mca_analyze(&h,&cfg,&cal,&w,&r)==MCA_OK);
        if (r.peak_count != 3U) { printf("Scenario duration=%u noise=%g seed=%u peaks=%u\n",
            (unsigned)durations[a],noises[b],seed,(unsigned)r.peak_count); }
        if(r.peak_count!=3U) { for(size_t q=0;q<r.peak_count;++q) printf("peak %u max=%u centroid=%g height=%g bg=%g\n",(unsigned)q,r.peaks[q].maximum_channel,r.peaks[q].centroid_channel,w.net[r.peaks[q].maximum_channel],w.background[r.peaks[q].maximum_channel]); } CHECK(r.peak_count==3U);
        for(size_t k=0;k<3;++k) {
            CHECK(fabs(r.peaks[k].centroid_channel-sim.peaks[k].channel)<2.5);
            CHECK(r.peaks[k].fwhm_valid);
        }
        ++scenarios;
    }
    /* Exact opening oracle over different window sizes, including boundaries. */
    const uint16_t radii[]={3U,20U,96U,1024U,2047U};
    static float erosion[MCA_CHANNELS];
    for(size_t k=0;k<5;++k) {
        cfg.background_radius=radii[k];
        for(size_t i=0;i<MCA_CHANNELS;++i) { h.spectrum[i]=(uint32_t)((i*2654435761U)^(i>>3)); }
        CHECK(mca_analyze(&h,&cfg,&cal,&w,&r)==MCA_OK);
        for(size_t i=0;i<MCA_CHANNELS;++i) {
            size_t lo=i>radii[k]?i-radii[k]:0U, hi=i+radii[k]<MCA_CHANNELS?i+radii[k]:MCA_CHANNELS-1U;
            float v=FLT_MAX;
            for(size_t j=lo;j<=hi;++j) { if(w.smooth[j]<v) { v=w.smooth[j]; } }
            erosion[i]=v;
        }
        for(size_t i=0;i<MCA_CHANNELS;++i) {
            size_t lo=i>radii[k]?i-radii[k]:0U, hi=i+radii[k]<MCA_CHANNELS?i+radii[k]:MCA_CHANNELS-1U;
            float v=0.0f;
            for(size_t j=lo;j<=hi;++j) { if(erosion[j]>v) { v=erosion[j]; } }
            CHECK(w.background[i]==v);
        }
    }
    /* Independent true five-peak spectrum: not a hard-coded three-peak filter. */
    cfg=mca_default_analysis_config();
    for(size_t i=0;i<MCA_CHANNELS;++i) {
        double count=20.0;
        for(size_t k=0;k<5;++k) { double z=((double)i-(200.0+350.0*k))/10.0; count+=(200.0+100.0*k)*exp(-0.5*z*z); }
        h.spectrum[i]=(uint32_t)(count+0.5);
    }
    CHECK(mca_analyze(&h,&cfg,&cal,&w,&r)==MCA_OK && r.peak_count==5U);
    for(size_t k=0;k<5;++k) { CHECK(fabs(r.peaks[k].centroid_channel-(200.0+350.0*k))<0.1); }
    printf("PASS phase2: %u simulated scenarios\n",scenarios);
}
int main(void)
{
    histogram_tests(); analysis_tests(); simulation_tests(); frame_tests(); phase2_tests();
    printf("ALL PASS: %u checks (host execution, no target hardware)\n", checks); return 0;
}


