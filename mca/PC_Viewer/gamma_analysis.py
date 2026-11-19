"""Offline K/eU/eTh analysis. NumPy only; no hardware or network side effects.

Concentrations are meaningful only for certified standards, matched geometry,
valid live times and decay-chain assumptions. Statistical errors below exclude
calibration, geometry, drift, template and background systematics.
"""
from dataclasses import dataclass
import csv
import itertools
from pathlib import Path
import numpy as np

ELEMENTS = ('K_percent', 'eU_ppm', 'eTh_ppm')
WINDOWS_KEV = ((1370., 1570.), (1660., 1860.), (2410., 2810.))


def _array(value, ndim=1):
    a = np.array(value, dtype=float, copy=True)
    if a.ndim != ndim or not a.size or not np.all(np.isfinite(a)):
        raise ValueError('Expected nonempty finite numeric array')
    return a


@dataclass(frozen=True)
class Spectrum:
    counts: np.ndarray
    live_time_s: float | None = None
    geometry_id: str | None = None
    source: str = ''

    def __post_init__(self):
        a = _array(self.counts)
        if len(a) < 8 or np.any(a < 0) or np.any(a != np.floor(a)) or np.any(a > 2**53-1):
            raise ValueError('Counts must be nonnegative exact integers <= 2^53-1, >=8 bins')
        if self.live_time_s is not None:
            if not np.isfinite(self.live_time_s) or not 1e-6 <= self.live_time_s <= 1e12:
                raise ValueError('live_time_s must be finite and within 1 us .. 1e12 s')
        a.setflags(write=False)
        object.__setattr__(self, 'counts', a)


def read_csv(path, live_time_s=None, geometry_id=None, channels=2048):
    """Strict raw histogram loader. Never infer live time from a filename."""
    values = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['channel', 'cumulative_count']:
            raise ValueError('CSV header must be channel,cumulative_count')
        for i, row in enumerate(reader):
            try:
                if None in row or int(row['channel']) != i:
                    raise ValueError('Channels must be unique, contiguous and start at 0')
                count = int(row['cumulative_count'])
                if not 0 <= count <= 2**53-1:
                    raise ValueError('Count out of range')
                values.append(count)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{path}: invalid row {i+2}: {exc}') from exc
    if len(values) != channels:
        raise ValueError(f'Expected {channels} channels; got {len(values)}')
    return Spectrum(values, live_time_s, geometry_id, str(Path(path).resolve()))


@dataclass(frozen=True)
class EnergyCalibration:
    slope: float
    intercept: float
    anchor_min: float
    anchor_max: float
    residuals_keV: tuple

    def __post_init__(self):
        if not np.all(np.isfinite([self.slope, self.intercept, self.anchor_min, self.anchor_max])):
            raise ValueError('Invalid calibration')
        if self.slope <= 0 or self.anchor_min < 0 or self.anchor_max <= self.anchor_min:
            raise ValueError('Calibration requires positive gain and distinct anchors')

    def energy(self, channel):
        return self.slope * np.asarray(channel) + self.intercept


def fit_energy(channels, energies_keV, max_residual_keV=10.):
    x, y = _array(channels), _array(energies_keV)
    if x.shape != y.shape or len(x) < 2 or len(np.unique(x)) != len(x):
        raise ValueError('At least two distinct confirmed calibration peaks required')
    if np.any(x < 0) or np.any(y <= 0) or np.any(np.diff(y[np.argsort(x)]) <= 0):
        raise ValueError('Energies must increase with channel')
    if not np.isfinite(max_residual_keV) or max_residual_keV <= 0:
        raise ValueError('Residual tolerance must be positive')
    # Center before fitting to avoid poorly scaled intercept/gain columns.
    centered = x-x.mean()
    slope = float(centered @ (y-y.mean()) / (centered @ centered))
    intercept = float(y.mean()-slope*x.mean())
    residual = y-(slope*x+intercept)
    if np.max(np.abs(residual)) > max_residual_keV:
        raise ValueError('Linear calibration residual exceeds tolerance')
    return EnergyCalibration(slope, intercept, float(x.min()), float(x.max()), tuple(residual))


def peak_candidates(spectrum, min_distance=20, significance=5.):
    """Exploratory channel candidates, NOT nuclide identifications or concentrations."""
    if not isinstance(min_distance, int) or min_distance < 1 or not np.isfinite(significance) or significance <= 0:
        raise ValueError('Invalid candidate search parameters')
    y = spectrum.counts
    smooth = np.convolve(y, np.ones(5)/5, 'valid')
    found = []
    for k in range(1, len(smooth)-1):
        i = k+2
        if smooth[k] <= smooth[k-1] or smooth[k] < smooth[k+1]:
            continue
        left = smooth[max(0, k-40):k]
        right = smooth[k+1:min(len(smooth), k+41)]
        base = max(float(left.min()), float(right.min()))
        if smooth[k]-base >= significance*np.sqrt(base+1):
            found.append((i, float(smooth[k]-base)))
    chosen = []
    for i, height in sorted(found, key=lambda p: -p[1]):
        if all(abs(i-p['channel']) >= min_distance for p in chosen):
            chosen.append({'channel': i, 'prominence_counts': height})
    return sorted(chosen, key=lambda p: p['channel'])


def measure_peak(spectrum, lo, hi, edge_bins=5, calibration=None):
    """Isolated-peak moment + interpolated FWHM, with linear sideband baseline.

    ROI ends must be peak-free; overlapping peaks require a different model.
    Net area is signed; centroid uses positive residuals and is noise-biased at
    low counts. FWHM uses unsmoothed data. No fit uncertainty is claimed.
    """
    if any(not isinstance(v, int) for v in (lo, hi, edge_bins)):
        raise ValueError('ROI bounds must be integers')
    if not 0 <= lo < hi < len(spectrum.counts) or edge_bins < 2 or hi-lo+1 < 2*edge_bins+5:
        raise ValueError('ROI too small or outside spectrum')
    x = np.arange(lo, hi+1, dtype=float)
    y = spectrum.counts[lo:hi+1]
    edges = np.r_[np.arange(edge_bins), np.arange(len(x)-edge_bins, len(x))]
    design = np.column_stack((np.ones(len(x)), x-x.mean()))
    baseline = design @ np.linalg.lstsq(design[edges], y[edges], rcond=None)[0]
    net = y-baseline
    positive = np.maximum(net, 0)
    m = int(np.argmax(net))
    if positive.sum() == 0 or net[m] < 5*np.sqrt(max(baseline[m], 0)+1):
        raise ValueError('No significant isolated peak in ROI')
    if m < edge_bins or m >= len(x)-edge_bins:
        raise ValueError('Peak intersects baseline sideband')
    centroid = float(x @ positive / positive.sum())
    l = r = m
    half = net[m]/2
    while l > 0 and net[l] > half:
        l -= 1
    while r < len(x)-1 and net[r] > half:
        r += 1
    width = None
    if l < m < r and net[l] <= half and net[r] <= half:
        xl = x[l]+(half-net[l])/(net[l+1]-net[l])
        xr = x[r-1]+(net[r-1]-half)/(net[r-1]-net[r])
        width = float(xr-xl)
    energy = float(calibration.energy(centroid)) if calibration else None
    width_keV = width*calibration.slope if width is not None and calibration else None
    return {'centroid_channel': centroid, 'fwhm_channel': width,
            'net_area_counts': float(net.sum()), 'energy_keV': energy,
            'fwhm_keV': width_keV,
            'resolution_percent': 100*width_keV/energy if width_keV is not None and energy > 0 else None,
            'method': 'isolated ROI, linear sidebands, positive moments, unsmoothed FWHM'}


def _well_conditioned(matrix, limit=1e6):
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0):
        raise ValueError('Missing response component')
    normalized = matrix/norms
    condition = float(np.linalg.cond(normalized))
    if np.linalg.matrix_rank(normalized) < 3 or condition > limit:
        raise ValueError('Rank-deficient or ill-conditioned calibration/response')
    return normalized, norms, condition


def nnls3(matrix, target):
    """Exact small-dimensional NNLS by all 8 active sets, no SciPy dependency."""
    a, b = _array(matrix, 2), _array(target)
    if a.shape != (len(b), 3):
        raise ValueError('Expected N x 3 response and N observations')
    a, scales, _ = _well_conditioned(a)
    best, error = np.zeros(3), float(b @ b)
    for n in (1, 2, 3):
        for active in itertools.combinations(range(3), n):
            coef = np.linalg.lstsq(a[:, active], b, rcond=None)[0]
            if np.any(coef < 0):
                continue
            residual = b-a[:, active] @ coef
            loss = float(residual @ residual)
            if loss < error:
                best = np.zeros(3)
                best[list(active)] = coef
                error = loss
    return best/scales


@dataclass(frozen=True)
class ResponseModel:
    response: np.ndarray
    transform: np.ndarray
    calibration: EnergyCalibration
    geometry_id: str
    method: str
    standard_condition: float


def _compatible(spectra):
    if not spectra or any(s.live_time_s is None for s in spectra):
        raise ValueError('Quantification requires measured live_time_s, not inferred wall time')
    geometry = spectra[0].geometry_id
    if not isinstance(geometry, str) or not geometry.strip():
        raise ValueError('Explicit matched geometry_id required')
    if any(s.geometry_id != geometry or len(s.counts) != len(spectra[0].counts) for s in spectra):
        raise ValueError('Geometry/channel mismatch; recalibration required')


def feature_transform(n, calibration, method, energy_range_keV=None):
    energies = calibration.energy(np.arange(n))
    if method == 'windows':
        rows = []
        for lo, hi in WINDOWS_KEV:
            if calibration.energy(-.5) > lo or calibration.energy(n-.5) < hi:
                raise ValueError('Energy coverage does not include complete K/U/Th windows')
            # Whole-bin integration preserves independent Poisson window counts.
            rows.append(((energies >= lo) & (energies < hi)).astype(float))
        transform = np.array(rows)
        if np.any(transform.sum(axis=1) < 2):
            raise ValueError('Insufficient energy sampling in a window')
        return transform
    if method != 'fsa' or energy_range_keV is None:
        raise ValueError('FSA requires an explicit trusted energy_range_keV')
    lo, hi = energy_range_keV
    if not np.isfinite(lo+hi) or lo >= hi or lo < energies[0] or hi > energies[-1]:
        raise ValueError('Invalid FSA energy range or incomplete coverage')
    selected = np.flatnonzero((energies >= lo) & (energies <= hi))
    if len(selected) < 4:
        raise ValueError('FSA range too narrow')
    return np.eye(n)[selected]


def calibrate_response(standards, concentrations, background, calibration,
                       method='windows', energy_range_keV=None):
    """Columns are K mass %, eU ppm, eTh ppm; mixed certified standards allowed.

    All spectra must already share the same energy/channel mapping. geometry_id
    is a user assertion, not an automatic verification of laboratory conditions.
    """
    _compatible([background, *standards])
    c = _array(concentrations, 2)
    if c.shape != (len(standards), 3) or len(standards) < 3 or np.any(c < 0):
        raise ValueError('Need >=3 certified standards and complete K/eU/eTh concentrations')
    _, _, condition = _well_conditioned(c)
    transform = feature_transform(len(background.counts), calibration, method, energy_range_keV)
    bg = background.counts/background.live_time_s
    rates = np.array([transform @ (s.counts/s.live_time_s-bg) for s in standards])
    # Nonnegative physical response estimate. Finite template errors are not propagated.
    response = np.array([nnls3(c, rates[:, i]) for i in range(rates.shape[1])])
    _well_conditioned(response)
    response.setflags(write=False)
    transform.setflags(write=False)
    return ResponseModel(response, transform, calibration, background.geometry_id, method, condition)


def quantify(spectrum, background, model):
    _compatible([spectrum, background])
    if spectrum.geometry_id != model.geometry_id or len(spectrum.counts) != model.transform.shape[1]:
        raise ValueError('Spectrum does not match calibrated response')
    t, tb = spectrum.live_time_s, background.live_time_s
    transform = model.transform
    y = transform @ (spectrum.counts/t-background.counts/tb)
    variance = (transform**2) @ (spectrum.counts/t**2+background.counts/tb**2)
    # Positive floor for empty bins; approximate weighted LS, not a Poisson fit.
    variance = np.maximum(variance, 1/t**2+1/tb**2)
    weighted = model.response/np.sqrt(variance[:, None])
    target = y/np.sqrt(variance)
    coef = nnls3(weighted, target)
    predicted = model.response @ coef
    residual = y-predicted
    active = coef > 0
    # Conditional Gaussian errors only for an interior solution. Boundary cases
    # need profile intervals, not a zero error or an asserted detection limit.
    errors = None
    if np.all(active):
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        covariance = (vt.T * (1/singular**2)) @ vt
        errors = dict(zip(ELEMENTS, np.sqrt(np.diag(covariance)).tolist()))
    flags = ['EQUIVALENT_U_TH_ASSUME_CHAIN_EQUILIBRIUM',
             'STATISTICAL_ONLY_EXCLUDES_TEMPLATE_AND_SYSTEMATIC_ERRORS',
             'LOW_COUNTS_REQUIRE_POISSON_OR_LONGER_ACQUISITION',
             'SAME_ENERGY_MAPPING_AND_GEOMETRY_ASSERTED_BY_USER']
    if not np.all(active):
        flags.append('BOUNDARY_SOLUTION_NO_DETECTION_LIMIT_OR_SYMMETRIC_ERROR')
    if model.method == 'windows':
        flags.append('THREE_WINDOWS_THREE_COMPONENTS_LIMIT_RESIDUAL_DIAGNOSTICS')
    used = np.flatnonzero(np.any(transform != 0, axis=0))
    if used.min() < model.calibration.anchor_min or used.max() > model.calibration.anchor_max:
        flags.append('ENERGY_MAPPING_EXTRAPOLATES_BEYOND_ANCHORS')
    return {'method': model.method, 'concentrations': dict(zip(ELEMENTS, coef.tolist())),
            'conditional_statistical_sigma': errors, 'quality_flags': flags,
            'weighted_residual_sum_squares': float((residual**2/variance).sum()),
            'nominal_dof': int(len(y)-np.count_nonzero(active)),
            'response_condition_scaled': _well_conditioned(model.response)[2],
            'net_rates': y.tolist(), 'fitted_rates': predicted.tolist(),
            'residual_rates': residual.tolist(), 'status': 'ESTIMATE_REQUIRES_VALIDATED_CALIBRATION'}
