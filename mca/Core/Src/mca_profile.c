#include "mca_profile.h"
#if defined(__arm__) || defined(__thumb__)
#include "stm32h7xx.h"
#endif
volatile mca_timing_t g_mca_timing;
void mca_profile_init(void)
{
#if defined(__arm__) || defined(__thumb__)
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55U;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    g_mca_timing.cpu_hz = SystemCoreClock;
#endif
}
uint32_t mca_cycles(void)
{
#if defined(__arm__) || defined(__thumb__)
    return DWT->CYCCNT;
#else
    return 0U;
#endif
}

#if defined(__arm__) || defined(__thumb__)
__attribute__((weak)) void mca_service(void) { }
#elif !defined(MCA_SERVICE_EXTERNAL)
void mca_service(void) { }
#endif

