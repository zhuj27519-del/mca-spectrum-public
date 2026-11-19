"""Deterministic Phase 2 simulation; no physical ADC data."""
import numpy as np
from protocol import Frame

def simulate(duration_ms=1000, sequence=0, seed=0x7432048, noise=1.0):
    if duration_ms <= 0:
        raise ValueError('Positive duration required')
    x = np.arange(2048, dtype=float)
    rate = 8 + .002*x
    for center, sigma, amplitude in [(400, 10, 800), (1000, 16, 1200), (1600, 22, 650)]:
        rate += amplitude*np.exp(-.5*((x-center)/sigma)**2)
    v = np.arange(2048, dtype=np.uint32) ^ np.uint32(seed)
    v ^= v >> 16
    v *= np.uint32(0x7feb352d)
    v ^= v >> 15
    v *= np.uint32(0x846ca68b)
    v ^= v >> 16
    mean = rate*(duration_ms/1000)
    values = np.maximum(0, mean + (2*v.astype(float)/4294967295-1)*np.sqrt(3)*noise*np.sqrt(mean))
    saturated = bool(np.any(values >= 4294967295-.5))
    bins = np.minimum(4294967295, np.floor(values+.5)).astype(np.uint32)
    return Frame(sequence, duration_ms, bins, saturated)
