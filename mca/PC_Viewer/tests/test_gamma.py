from pathlib import Path
import json
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gamma_analysis import (Spectrum, read_csv, fit_energy, measure_peak, nnls3,
                            calibrate_response, quantify, peak_candidates, feature_transform)
from gamma_cli import generate_demo, run_manifest, main


def test_energy_and_nonzero_intercept_resolution():
    cal = fit_energy([100, 400, 900], [210, 810, 1810])
    assert cal.slope == pytest.approx(2)
    assert cal.intercept == pytest.approx(10)
    x = np.arange(2048)
    s = Spectrum(np.rint(30+.02*x+10000*np.exp(-.5*((x-400)/10)**2)))
    p = measure_peak(s, 340, 460, calibration=cal)
    assert p['centroid_channel'] == pytest.approx(400, abs=.01)
    assert p['fwhm_channel'] == pytest.approx(2*np.sqrt(2*np.log(2))*10, abs=.03)
    assert p['resolution_percent'] == pytest.approx(100*p['fwhm_keV']/810, abs=.001)
    assert p['resolution_percent'] != pytest.approx(100*p['fwhm_channel']/400, abs=.001)


@pytest.mark.parametrize('x,y', [([1],[662]), ([1,1],[662,1461]), ([1,2],[1461,662]),
                                ([-1,2],[10,20]), ([1,2],[10,np.nan]),
                                ([1,2,3],[10,20,1000])])
def test_bad_calibration(x, y):
    with pytest.raises(ValueError):
        fit_energy(x, y)


@pytest.mark.parametrize('counts,time', [([1]*7,None), ([-1]*8,None), ([.1]*8,None),
                                       ([np.nan]*8,None), ([2**53]*8,None), ([1]*8,0),
                                       ([1]*8,-1), ([1]*8,np.inf)])
def test_invalid_spectrum(counts, time):
    with pytest.raises(ValueError):
        Spectrum(counts,time)


def test_csv_validation(tmp_path):
    p = tmp_path/'s.csv'
    p.write_text('channel,cumulative_count\n'+'\n'.join(f'{i},0' for i in range(8)))
    s = read_csv(p, channels=8)
    assert s.live_time_s is None
    assert s.counts.sum() == 0
    with pytest.raises(ValueError):
        read_csv(p)
    p.write_text('channel,cumulative_count\n0,1\n0,2')
    with pytest.raises(ValueError):
        read_csv(p, channels=8)
    p.write_text('channel,cumulative_count\n0,1.5')
    with pytest.raises(ValueError):
        read_csv(p, channels=8)


def test_nnls_is_not_postfit_clipping():
    a = np.array([[1,1,0],[0,1,0],[0,0,1],[1,0,1]], float)
    b = a @ [-2,3,4]
    got = nnls3(a,b)
    clipped = np.maximum(np.linalg.lstsq(a,b,rcond=None)[0],0)
    assert np.linalg.norm(a @ got-b) < np.linalg.norm(a @ clipped-b)
    assert np.all(got >= 0)
    grad = a.T @ (a @ got-b)
    assert np.all(grad[got == 0] >= -1e-10)
    np.testing.assert_allclose(grad[got > 0],0,atol=1e-10)


def test_nnls_units_and_rank():
    a = np.array([[1,2,1],[2,1,3],[3,4,1],[2,3,4]],float)*[1e-3,1,1e3]
    np.testing.assert_allclose(nnls3(a, a @ [2,5,7]),[2,5,7], rtol=1e-7)
    with pytest.raises(ValueError):
        nnls3(np.ones((10,3)),np.ones(10))


def test_nnls_random_kkt_conditions():
    rng = np.random.default_rng(2048)
    for _ in range(80):
        a = rng.uniform(.01, 2, (30, 3))
        b = a @ rng.uniform(-3, 4, 3)+rng.normal(0,.2,30)
        x = nnls3(a,b)
        gradient = a.T @ (a @ x-b)
        assert np.all(x >= 0)
        assert np.all(gradient[x == 0] >= -1e-8)
        np.testing.assert_allclose(gradient[x > 0],0,atol=1e-8)


def test_snapshot_is_owned_and_extreme_duration_rejected():
    counts = np.ones(2048)
    s = Spectrum(counts,100,'lab')
    counts[0] = 200
    assert s.counts[0] == 1
    with pytest.raises(ValueError):
        s.counts[0] = 2
    with pytest.raises(ValueError):
        Spectrum(counts,1e-200,'lab')


def test_demo_mixed_standards_and_cli(tmp_path):
    folder = tmp_path/'demo'
    generate_demo(folder)
    result = run_manifest(folder/'manifest.json')
    for method in ('windows','fsa'):
        a = result['analysis'][method]
        np.testing.assert_allclose(list(a['concentrations'].values()),[1,3,5], atol=1e-8)
        assert a['weighted_residual_sum_squares'] < 1e-12
    output = tmp_path/'result.json'
    assert main(['run',str(folder/'manifest.json'),'--output',str(output)]) == 0
    assert json.loads(output.read_text())['manifest_sha256']


def fixture_model(method='windows'):
    x = np.arange(2048)
    response = np.column_stack([np.rint(200*np.exp(-.5*((x-m)/s)**2))
                               for m,s in [(730,15),(882,20),(1307,30)]])
    response[:,1] += (x < 950)*2
    response[:,2] += (x < 1400)*3
    bg = Spectrum(np.full(2048,400),100,'lab')
    standard_c = np.array([[2,1,1],[1,6,1],[1,1,10]])
    standards = [Spectrum((4+response @ c)*100,100,'lab') for c in standard_c]
    cal = fit_energy([300,1400],[600,2800])
    model = calibrate_response(standards,standard_c,bg,cal,method,(400,2800))
    return model, bg, response, standards, standard_c


@pytest.mark.parametrize('method',['windows','fsa'])
def test_duration_scaling_noise_and_background(method):
    model,bg,response,_,_ = fixture_model(method)
    truth = np.array([2.,3.,6.])
    sample = Spectrum((4+response @ truth)*50,50,'lab')
    a = quantify(sample,bg,model)
    np.testing.assert_allclose(list(a['concentrations'].values()),truth,atol=1e-8)
    rng = np.random.default_rng(743)
    sample = Spectrum(rng.poisson((4+response @ truth)*50),50,'lab')
    estimates = quantify(sample,bg,model)
    np.testing.assert_allclose(list(estimates['concentrations'].values()),truth,rtol=.02)
    assert all(s > 0 for s in estimates['conditional_statistical_sigma'].values())


def test_missing_metadata_and_rank():
    model,bg,response,standards,c = fixture_model()
    for s in [Spectrum(bg.counts),Spectrum(bg.counts,100,'other')]:
        with pytest.raises(ValueError):
            quantify(s,bg,model)
    with pytest.raises(ValueError):
        calibrate_response(standards,np.ones((3,3)),bg,model.calibration)
    with pytest.raises(ValueError):
        feature_transform(2048,fit_energy([0,1000],[1,1001]),'windows')


def test_zero_and_negative_net_counts_are_not_detections():
    model,bg,_,_,_ = fixture_model()
    for sample in [bg,Spectrum(np.zeros(2048),100,'lab')]:
        result = quantify(sample,bg,model)
        assert result['concentrations'] == dict(K_percent=0.,eU_ppm=0.,eTh_ppm=0.)
        assert result['conditional_statistical_sigma'] is None
        assert any('BOUNDARY' in f for f in result['quality_flags'])


def test_peak_edges_and_flat():
    s = Spectrum(np.zeros(2048))
    assert peak_candidates(s) == []
    for lo,hi in [(-1,20),(0,2048),(10,12),(100,200)]:
        with pytest.raises(ValueError):
            measure_peak(s,lo,hi)


def test_manifest_gates_and_uncalibrated_mode(tmp_path):
    folder = tmp_path/'demo'
    generate_demo(folder)
    p = folder/'manifest.json'
    original = json.loads(p.read_text())
    for field in ('certified_concentrations_confirmed','same_energy_mapping_confirmed'):
        config = dict(original); config[field] = False
        p.write_text(json.dumps(config))
        with pytest.raises(ValueError):
            run_manifest(p)
    original['sample']['live_time_verified'] = False
    p.write_text(json.dumps(original))
    assert main(['run',str(p)]) == 2
    original.pop('standards')
    p.write_text(json.dumps(original))
    result = run_manifest(p)
    assert result['concentrations'] is None
    assert 'NO_CONCENTRATION' in result['status']
