"""python gamma_cli.py inspect spectrum.csv; python gamma_cli.py run manifest.json."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from k40 import LINE_KEV, analyze as analyze_k40
from gamma_analysis import (Spectrum, read_csv, fit_energy, measure_peak,
                            peak_candidates, calibrate_response, quantify)


def _load(base, item):
    path = (base / item['csv']).resolve()
    spectrum = read_csv(path, item.get('live_time_s'), item.get('geometry_id'))
    return spectrum, {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'live_time_s': spectrum.live_time_s, 'geometry_id': spectrum.geometry_id}


def run_manifest(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if config.get('schema_version') != 1:
        raise ValueError('Expected manifest schema_version 1')
    points = config['energy_points']
    calibration = fit_energy([p['channel'] for p in points], [p['keV'] for p in points],
                             config.get('max_residual_keV', 10.))
    sample, provenance = _load(path.parent, config['sample'])
    if calibration.anchor_max > len(sample.counts)-1:
        raise ValueError('Energy anchor outside acquired channels')
    result = {'schema_version': 1, 'energy_calibration': asdict(calibration),
              'sample': provenance, 'total_counts': sum(map(int, sample.counts)),
              'peaks': [measure_peak(sample, p['lo'], p['hi'], p.get('edge_bins', 5), calibration)
                        for p in config.get('peak_rois', [])],
              'concentrations': None,
              'status': 'ENERGY_CALIBRATED_ONLY_NO_CONCENTRATION_CALIBRATION'}
    standards = config.get('standards', [])
    if not standards:
        return result
    if config.get('certified_concentrations_confirmed') is not True:
        raise ValueError('Confirm certified concentration units K_percent/eU_ppm/eTh_ppm')
    if config.get('same_energy_mapping_confirmed') is not True:
        raise ValueError('Confirm identical energy/channel mapping for all spectra')
    entries = [config['sample'], config['background'], *standards]
    if any(e.get('live_time_verified') is not True for e in entries):
        raise ValueError('Every input requires measured live_time_s and live_time_verified=true')
    background, bg_source = _load(path.parent, config['background'])
    loaded = [_load(path.parent, s) for s in standards]
    concentrations = [[s[k] for k in ('K_percent', 'eU_ppm', 'eTh_ppm')] for s in standards]
    method = config.get('method', 'windows')
    methods = ('windows', 'fsa') if method == 'both' else (method,)
    result['analysis'] = {}
    for selected in methods:
        model = calibrate_response([s for s, _ in loaded], concentrations, background, calibration,
                                   selected, config.get('energy_range_keV'))
        result['analysis'][selected] = quantify(sample, background, model)
    result.update(status='QUANTIFIED_WITH_USER_SUPPLIED_CALIBRATION',
                  background=bg_source, standards=[p for _, p in loaded])
    # Include the calibration manifest for reproducibility, not just the sample hash.
    result['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['calibration_inputs'] = config
    result.pop('concentrations')
    return result


def run_k40(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    if cfg.get('schema_version') != 1:
        raise ValueError('Expected schema_version 1')
    n = cfg.get('channels', 2048)
    if type(n) is not int or n < 8:
        raise ValueError('Invalid channel count')
    if 'energy_keV' not in cfg:
        points = cfg['energy_points']
        cal = fit_energy([p['channel'] for p in points], [p['keV'] for p in points],
                         cfg.get('max_residual_keV', 10.))
        if cal.anchor_max >= n:
            raise ValueError('Energy anchor outside acquired spectrum')
        span = 3 * cfg['fwhm_keV']
        if not cal.energy(cal.anchor_min) <= LINE_KEV-span < LINE_KEV+span <= cal.energy(cal.anchor_max):
            raise ValueError('Energy anchors must bracket the K-40 peak and sidebands')
        cfg['energy_keV'] = cal.energy(np.arange(n))
    elif 'energy_points' in cfg or len(cfg['energy_keV']) != n:
        raise ValueError('Provide one calibrated energy axis matching the channel count')
    paths = [(path.parent / cfg[key]['csv']).resolve()
             for key in ('sample', 'background', 'standard') if key in cfg]
    if len(paths) != len(set(paths)):
        raise ValueError('Sample, background and standard must be independent acquisitions')
    hashes = {}
    def load(entry):
        csv_path = (path.parent / entry['csv']).resolve()
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if digest in hashes.values():
            raise ValueError('Duplicate acquisition contents')
        hashes[entry['csv']] = digest
        return read_csv(csv_path, channels=n).counts
    result = analyze_k40(cfg, load)
    result.update(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  inputs_sha256=hashes, fwhm_keV=cfg['fwhm_keV'])
    result['synthetic_demo_only'] = cfg.get('synthetic_demo_only') is True
    return result


def generate_demo(folder):
    """Artificial response mixtures; these are NOT detector calibration constants."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    channel = np.arange(2048)
    response = np.column_stack([np.rint(200*np.exp(-.5*((channel-m)/s)**2))
                                for m, s in [(731, 12), (882, 15), (1307, 20)]])
    # Deliberate cross-window continuum so the test is not just three isolated ROIs.
    response[:, 1] += (channel < 900)*2
    response[:, 2] += (channel < 1400)*3
    background = np.full(2048, 4)
    certificates = [[2, 1, 1], [1, 8, 2], [1, 2, 12]]
    truth = [1, 3, 5]
    def save(name, counts):
        np.savetxt(folder/name, np.column_stack((channel, counts)), delimiter=',', fmt='%d',
                   header='channel,cumulative_count', comments='')
        return {'csv': name, 'live_time_s': 100., 'live_time_verified': True,
                'geometry_id': 'SYNTHETIC_ONLY_NOT_LAB_CALIBRATION'}
    bg = save('background.csv', background*100)
    sample = save('sample.csv', (background+response @ truth)*100)
    standards = []
    for i, c in enumerate(certificates):
        spec = save(f'standard_{i}.csv', (background+response @ c)*100)
        spec.update(dict(zip(('K_percent', 'eU_ppm', 'eTh_ppm'), c)))
        standards.append(spec)
    config = {'schema_version': 1, 'synthetic_demo_only': True,
              'certified_concentrations_confirmed': True, 'same_energy_mapping_confirmed': True,
              'energy_points': [{'channel': 300., 'keV': 600.}, {'channel': 1400., 'keV': 2800.}],
              'sample': sample, 'background': bg, 'standards': standards,
              'method': 'both', 'energy_range_keV': [400., 2800.],
              'expected_synthetic_concentrations': truth}
    (folder/'manifest.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return {'demo_manifest': str((folder/'manifest.json').resolve()), 'synthetic_only': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    inspect = subs.add_parser('inspect', help='Raw channel candidates only; no nuclide assignment')
    inspect.add_argument('csv', nargs='+')
    inspect.add_argument('--output')
    run = subs.add_parser('run', help='Energy/response analysis using explicit manifest')
    run.add_argument('manifest')
    run.add_argument('--output')
    k40 = subs.add_parser('k40', help='Check the K-40 line and optionally estimate Bq/kg')
    k40.add_argument('manifest')
    k40.add_argument('--output')
    demo = subs.add_parser('demo', help='Create synthetic fixtures in a NEW directory')
    demo.add_argument('directory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            result = []
            for path in args.csv:
                s = read_csv(path)
                result.append({'source': s.source, 'channels': len(s.counts),
                               'total_counts': sum(map(int, s.counts)),
                               'candidate_peaks': peak_candidates(s), 'concentrations': None,
                               'status': 'UNCALIBRATED_CHANNEL_CANDIDATES_ONLY'})
        elif args.command == 'k40':
            result = run_k40(args.manifest)
        elif args.command == 'demo':
            result = generate_demo(args.directory)
        else:
            result = run_manifest(args.manifest)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if getattr(args, 'output', None):
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload+'\n', encoding='utf-8')
            print(str(output.resolve()))
        else:
            print(payload)
        return 0
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
