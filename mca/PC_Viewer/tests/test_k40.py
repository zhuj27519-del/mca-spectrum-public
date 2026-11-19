import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gamma_cli import main, run_k40
from k40 import check_line, photopeak


class K40Test(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.energy = np.arange(2048, dtype=float)
        line = np.rint(400*np.exp(-.5*((self.energy-1460.8)/10)**2)).astype(int)
        self.entries = {}
        for name, time, amplitude in [('sample', 30, 2), ('background', 60, 0), ('standard', 15, 4)]:
            counts = time * (12 + amplitude*line)
            np.savetxt(self.folder/(name+'.csv'), np.column_stack([self.energy, counts]),
                       fmt='%d', delimiter=',', header='channel,cumulative_count', comments='')
            self.entries[name] = dict(csv=name+'.csv', live_time_s=time, live_time_verified=True)
        self.config = dict(schema_version=1, fwhm_keV=24, geometry_id='fixture',
                           matched_acquisitions_confirmed=True, standard_response_applicable=True,
                           energy_points=[dict(channel=500, keV=500), dict(channel=1800, keV=1800)],
                           **self.entries)
        self.config['sample']['mass_kg'] = .5
        self.config['standard'].update(mass_kg=.75, K40_Bq_kg=200)

    def manifest(self):
        path = self.folder/'manifest.json'
        path.write_text(json.dumps(self.config))
        return path

    def test_matched_standard_cli(self):
        path = self.manifest()
        output = self.folder/'result.json'
        self.assertEqual(main(['k40', str(path), '--output', str(output)]), 0)
        result = json.loads(output.read_text())
        self.assertAlmostEqual(result['activity_Bq_kg'], 150)
        self.assertGreater(result['counting_sigma_Bq_kg'], 0)
        self.assertEqual(len(result['inputs_sha256']), 3)

    def test_gross_without_standard(self):
        del self.config['standard'], self.config['background']
        r = run_k40(self.manifest())
        self.assertEqual(r['status'], 'GROSS_LINE_CANDIDATE')
        self.assertIsNone(r['activity_Bq_kg'])

    def test_flat_sample(self):
        self.config['sample']['csv'] = 'flat.csv'
        np.savetxt(self.folder/'flat.csv', np.column_stack([self.energy, self.energy*0+360]),
                   fmt='%d', delimiter=',', header='channel,cumulative_count', comments='')
        result = run_k40(self.manifest())
        self.assertEqual(result['status'], 'NO_CLEAR_SAMPLE_EXCESS')
        self.assertIsNone(result['activity_Bq_kg'])

    def test_do_not_extrapolate(self):
        self.config['energy_points'][1] = dict(channel=1000, keV=1000)
        with self.assertRaisesRegex(ValueError, 'bracket'):
            run_k40(self.manifest())

    def test_no_duplicate_blank(self):
        self.config['sample']['csv'] = self.config['background']['csv']
        with self.assertRaisesRegex(ValueError, 'independent'):
            run_k40(self.manifest())

    def test_requires_live_time(self):
        self.config['background']['live_time_verified'] = False
        with self.assertRaisesRegex(ValueError, 'live times'):
            run_k40(self.manifest())

    def test_wrong_peak_offset(self):
        y = np.rint(15 + 1000*np.exp(-.5*((self.energy-1479)/4)**2))
        self.assertFalse(check_line(self.energy, y, 24)['candidate'])

    def test_empty_window_and_linear_continuum(self):
        self.assertAlmostEqual(photopeak(self.energy, 100+self.energy, 24)[0], 0, places=8)
        self.assertFalse(check_line(self.energy, np.zeros(2048), 24)['candidate'])
        with self.assertRaisesRegex(ValueError, 'cover'):
            photopeak(self.energy[:1400], np.zeros(1400), 24)


if __name__ == '__main__':
    unittest.main()
