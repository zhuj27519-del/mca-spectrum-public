"""Targeted K-40 photopeak analysis of raw counts on a calibrated energy axis."""
import numpy as np

LINE_KEV = 1460.8


def peak_weights(energy, fwhm_keV):
    """Integrate ±1 FWHM with a linear continuum interpolated from two sidebands."""
    energy = np.asarray(energy, dtype=float)
    if (energy.ndim != 1 or len(energy) < 8 or not np.isfinite(energy).all()
            or np.any(np.diff(energy) <= 0)):
        raise ValueError('Expected increasing finite bin centers in keV')
    if not np.isfinite(fwhm_keV) or not 0 < fwhm_keV < LINE_KEV / 3:
        raise ValueError('Expected positive detector FWHM in keV at the K-40 line')
    z = (energy - LINE_KEV) / fwhm_keV
    if z[0] > -3 or z[-1] < 3:
        raise ValueError('Energy axis must cover the peak and both sidebands')
    left, peak, right = (z >= -3) & (z <= -2), abs(z) <= 1, (z >= 2) & (z <= 3)
    if min(left.sum(), right.sum()) < 2 or peak.sum() < 5:
        raise ValueError('Insufficient energy bins for the specified resolution')
    # Sideband counts estimate a continuum density (counts/keV).
    widths = np.gradient(energy)
    def center(mask):
        return np.average(energy[mask], weights=widths[mask])
    fraction = (center(peak) - center(left)) / (center(right) - center(left))
    weights = peak.astype(float)
    weights[left] = -(1-fraction) * widths[peak].sum() / widths[left].sum()
    weights[right] = -fraction * widths[peak].sum() / widths[right].sum()
    return weights, peak


def photopeak(energy, counts, fwhm_keV):
    counts = np.asarray(counts, dtype=float)
    weights, _ = peak_weights(energy, fwhm_keV)
    if (counts.shape != weights.shape or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts != np.floor(counts)).any()):
        raise ValueError('Expected one nonnegative integer raw count per energy bin')
    return float(weights @ counts), float((weights**2) @ counts)


def check_line(energy, counts, fwhm_keV):
    """Fixed single-line screen. The significance is a counting approximation."""
    energy, counts = np.asarray(energy), np.asarray(counts)
    area, var = photopeak(energy, counts, fwhm_keV)
    _, peak = peak_weights(energy, fwhm_keV)
    # Use a smoothed density for locating the maximum; area always uses raw bins.
    density = counts[peak] / np.gradient(energy)[peak]
    width = max(1, round(len(density) / 8))
    smooth = np.convolve(density, np.ones(width) / width, mode='valid')
    centers = np.convolve(energy[peak], np.ones(width) / width, mode='valid')
    maximum = float(centers[np.argmax(smooth)])
    z = area / np.sqrt(var) if var > 0 else 0.0
    candidate = bool(z >= 5 and abs(maximum - LINE_KEV) <= fwhm_keV / 2)
    return dict(candidate=candidate, maximum_keV=maximum, roi_area_counts=area,
                counting_sigma_counts=float(np.sqrt(var)), approximate_z=float(z))


def input_value(item, key):
    v = item[key]
    if isinstance(v, bool) or not np.isfinite(v) or v <= 0:
        raise ValueError(f'{key} must be finite and positive')
    return float(v)


def analyze(config, load_spectrum):
    """Compare sample and reference net rates against the same independent blank."""
    width = input_value(config, 'fwhm_keV')
    energy = config['energy_keV']
    sample = config['sample']
    counts = load_spectrum(sample)
    line = check_line(energy, counts, width)
    out = dict(nuclide='K-40', line_keV=LINE_KEV, line=line, activity_Bq_kg=None,
               status='GROSS_LINE_CANDIDATE' if line['candidate'] else 'NO_CLEAR_LINE')
    if 'background' not in config:
        if 'standard' in config:
            raise ValueError('Standard-based quantification requires a matched background')
        return out

    background = config['background']
    standard = config.get('standard')
    entries = [sample, background] + ([standard] if standard is not None else [])
    if (not config.get('geometry_id') or config.get('matched_acquisitions_confirmed') is not True
            or any(e.get('live_time_verified') is not True for e in entries)):
        raise ValueError('Confirm common geometry, detector, energy mapping and measured live times')
    def rate(entry, raw):
        t = input_value(entry, 'live_time_s')
        area, variance = photopeak(energy, raw, width)
        return area / t, variance / t**2
    bg_rate, bg_var = rate(background, load_spectrum(background))
    sample_rate, sample_var = rate(sample, counts)
    net = sample_rate - bg_rate
    sigma = np.sqrt(sample_var + bg_var)
    excess = bool(line['candidate'] and sigma > 0 and net >= 5 * sigma)
    out.update(net_roi_cps=net, counting_sigma_cps=float(sigma),
               status='SAMPLE_EXCESS_CANDIDATE' if excess else 'NO_CLEAR_SAMPLE_EXCESS')
    if standard is None:
        return out
    if config.get('standard_response_applicable') is not True:
        raise ValueError('Confirm certificate, matrix, filling geometry and matched ROI response')
    ref_counts = load_spectrum(standard)
    ref_rate, ref_var = rate(standard, ref_counts)
    ref_net = ref_rate - bg_rate
    if (not check_line(energy, ref_counts, width)['candidate']
            or ref_net <= 0 or ref_net < 5 * np.sqrt(ref_var + bg_var)):
        raise ValueError('Insufficient K-40 excess in calibration standard')
    scale = (input_value(standard, 'K40_Bq_kg') * input_value(standard, 'mass_kg')
             / input_value(sample, 'mass_kg'))
    if excess:
        ratio = net / ref_net
        # The same blank enters both differences. Include its covariance.
        variance = (sample_var + ratio**2 * ref_var + (ratio-1)**2 * bg_var) / ref_net**2
        out.update(status='STANDARD_BASED_ESTIMATE', activity_Bq_kg=scale*ratio,
                   uncertainty_scope='Counting only, conditional on calibration and geometry',
                   counting_sigma_Bq_kg=float(scale*np.sqrt(variance)))
    return out
