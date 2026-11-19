#ifndef MCA_PROFILE_H
#define MCA_PROFILE_H
#include <stdint.h>
typedef struct {
    uint32_t cpu_hz, simulation, background, detection, centroid, fwhm, other, total;
    uint32_t cache_build, samples;
} mca_timing_t;
extern volatile mca_timing_t g_mca_timing;
void mca_service(void); /* Bounded foreground network poll; weak no-op on host. */
void mca_profile_init(void);
uint32_t mca_cycles(void);
#endif
