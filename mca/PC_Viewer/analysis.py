"""PC recomputation: MCA1 carries counts/duration, not firmware analysis fields.
Matches the Phase 2 algorithm/defaults; values are channel quantities, not keV.
"""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class Peak:
    maximum: int
    centroid: float
    fwhm: float | None
    resolution: float | None

def extreme(values, radius, maximum=False):
    out = np.empty(len(values), dtype=np.float32)
    q, next_index = deque(), 0
    for i in range(len(values)):
        hi, lo = min(len(values)-1, i+radius), max(0, i-radius)
        while next_index <= hi:
            while q and ((values[q[-1]] <= values[next_index]) if maximum else (values[q[-1]] >= values[next_index])):
                q.pop()
            q.append(next_index)
            next_index += 1
        while q[0] < lo:
            q.popleft()
        out[i] = values[q[0]]
    return out

def analyze(bins):
    y = np.asarray(bins, dtype=np.float64)
    if y.shape != (2048,) or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError('Expected 2048 finite nonnegative counts')
    smooth = (np.convolve(y, np.ones(5), 'same') / np.convolve(np.ones(2048), np.ones(5), 'same')).astype(np.float32)
    background = extreme(extreme(smooth, 96), 96, True)
    net = np.maximum(0, smooth-background)
    quiet = net <= 5*np.sqrt(background+np.float32(1))
    mask = quiet[1:] & quiet[:-1]
    diffs = np.diff(y)-np.diff(background.astype(float))
    power = (y[1:]+y[:-1])[mask].sum()
    noise = max(1.0, math.sqrt(np.square(diffs[mask]).sum()/power)) if power else 1.0
    candidates = []
    for i in range(2048):
        value = float(net[i])
        if value < 20 or value < 5*math.sqrt(background[i]+1) or (i and value <= net[i-1]) or (i < 2047 and value < net[i+1]):
            continue
        left = right = value
        for j in range(i-1, -1, -1):
            if net[j] > value:
                break
            left = min(left, float(net[j]))
        for j in range(i+1, 2048):
            if net[j] > value:
                break
            right = min(right, float(net[j]))
        prominence = value - max(left if i else 0, right if i < 2047 else 0)
        if prominence >= max(15, .05*value, noise*5*math.sqrt(float(smooth[i])+1)):
            candidates.append(i)
    selected = []
    for i in sorted(candidates, key=lambda i: -net[i]):
        if all(abs(i-j) >= 60 for j in selected):
            selected.append(i)
            if len(selected) == 8:
                break
    selected.sort()
    peaks = []
    for k, m in enumerate(selected):
        lo, hi = max(0, m-80), min(2047, m+80)
        if k:
            lo = max(lo, (m+selected[k-1])//2+1)
        if k+1 < len(selected):
            hi = min(hi, (m+selected[k+1])//2)
        weights = np.maximum(0, y[lo:hi+1]-background[lo:hi+1])
        centroid = float(np.dot(np.arange(lo, hi+1), weights)/weights.sum()) if weights.sum() else float(m)
        half, l, r = float(net[m])*.5, m, m
        while l > lo and net[l] > half:
            l -= 1
        while r < hi and net[r] > half:
            r += 1
        width = None
        if l < m < r and net[l] <= half and net[r] <= half:
            left = l+(half-net[l])/(net[l+1]-net[l])
            right = r-1+(net[r-1]-half)/(net[r-1]-net[r])
            width = float(right-left)
        peaks.append(Peak(m, centroid, width, 100*width/centroid if width and centroid > 0 else None))
    return peaks, background
