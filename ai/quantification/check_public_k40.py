"""Reproduce two public K-40 checks without publishing third-party spectra."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mca' / 'PC_Viewer'))
from k40 import check_line

HTML_SHA256 = '91eac5159ca906945dee929e23e674b0c17e4a93726eeb7b43c525761e2814ab'


def open_gamma(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HTML_SHA256:
        raise ValueError('Unexpected OpenGamma HTML, see the pinned source in the README')
    # Decode the plotted arrays as JSON; the upstream JavaScript is never executed.
    payload = raw.decode().split("Plotly.newPlot('plotly-output',", 1)[1]
    traces = {v['name']: v for v in json.JSONDecoder().raw_decode(payload)[0]}
    bg, clean = traces['Background'], traces['Clean Spectrum']
    if bg['x'] != clean['x']:
        raise ValueError('Energy axes differ')
    gross = np.asarray(clean['y']) + bg['y']
    return dict(channels=len(gross), assumed_fwhm_keV=3.,
                gross=check_line(bg['x'], gross, 3.),
                background=check_line(bg['x'], bg['y'], 3.))


def published_rates():
    # Hoeling, Reed & Siegel, Am. J. Phys. 67, 440 (1999), Tables I and II.
    # sample, mass (g), before/gross/after (counts per minute),
    # sample-specific sensitivity (cpm/g natural K), paper's K mass percent.
    inputs = [('prune juice', 3102, 9.6, 17.6, 9.9, 1.13, .22),
              ('bananas', 2975, 9.9, 22.4, 10.5, 1.02, .40),
              ('potatoes', 3249, 10.7, 30.2, 9.9, 1.02, .59),
              ('soil', 3280, 9.7, 86.4, 10.2, 1.17, 2.00)]
    rows = []
    for name, mass, before, gross, after, sensitivity, reported in inputs:
        net = gross - (before + after) / 2
        percent = 100 * net / (sensitivity * mass)
        rows.append(dict(sample=name, potassium_mass_percent=percent,
                         paper_percent=reported, difference_percentage_points=percent-reported,
                         derived_K40_Bq_kg=313 * percent))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open-gamma-html', type=Path)
    args = parser.parse_args()
    result = dict(published_peak_rates=published_rates())
    if args.open_gamma_html:
        result['public_hpge_line'] = open_gamma(args.open_gamma_html)
    print(json.dumps(result, indent=2, allow_nan=False))
