"""Evaluate the fixed K-40 ROI on the pinned eight-sample LaBr3 workbook."""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mca' / 'PC_Viewer'))
from k40 import check_line, photopeak
from regression import EXPECTED_SHA


def evaluate(path, fwhm_keV):
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise ValueError('Unexpected workbook. Use Mendeley n6yfhrz824 version 1.')
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb['Spectra'].values)
        meta = list(wb['meta-data'].values)[5:13]
    finally:
        wb.close()
    energy = np.array([r[1] for r in rows[1:1025]], float)
    samples = []
    for row in meta:
        name, live, mass = row[:3]
        col = rows[0].index(name)
        counts = np.array([r[col] for r in rows[1:1025]], float)
        live, mass = float(live), float(mass)
        reference = float(str(row[7]).split('±')[0])
        if not np.isfinite([live, mass, reference]).all() or min(live, mass) <= 0 or reference < 0:
            raise ValueError(f'Invalid metadata for {name}')
        rate = photopeak(energy, counts, fwhm_keV)[0] / live
        samples.append((name, mass, reference, rate, check_line(energy, counts, fwhm_keV)))
    if len(samples) != 8 or len({r[0] for r in samples}) != 8:
        raise ValueError('Expected eight distinct named samples')
    # The native mass unit cancels between training and prediction.
    results = []
    for name, mass, reference, rate, line in samples:
        train = [s for s in samples if s[0] != name]
        design = np.array([[m * a, 1.] for _, m, a, _, _ in train])
        (slope, intercept), _, rank, _ = np.linalg.lstsq(design, [s[3] for s in train], rcond=None)
        if rank < 2 or slope <= 0:
            raise ValueError(f'Unusable training response in fold {name}')
        estimate = (rate-intercept) / (mass*slope)
        results.append(dict(sample=name, reference_Bq_kg=reference, estimate_Bq_kg=estimate,
                            error_Bq_kg=estimate-reference, gross_line_candidate=line['candidate'],
                            roi_cps=rate))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--fwhm-kev', type=float, required=True,
                        help='Detector resolution at 1460.8 keV, chosen before evaluation')
    args = parser.parse_args()
    results = evaluate(args.workbook, args.fwhm_kev)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
    print('Leave-one-sample-out MAE (Bq/kg):', np.mean([abs(r['error_Bq_kg']) for r in results]),
          file=sys.stderr)
