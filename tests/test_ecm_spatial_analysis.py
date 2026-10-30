import csv
import json
import os
import shutil
import tempfile
import unittest

from validation.analyze_ecm_spatial_anomalies import main, stream_field


class ECMSpatialAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sim = os.path.join(self.root, 'sim_output')
        os.makedirs(self.sim)
        with open(os.path.join(self.sim, 'data.csv'), 'w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['', 'time'])
            writer.writerows([[0, 0.0], [1, 1.0], [2, 2.0]])

    def tearDown(self):
        shutil.rmtree(self.root)

    def _field(self, name, rows):
        path = os.path.join(self.sim, name + '_data.csv')
        with open(path, 'w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['', '10', '20', 'time'])
            for index, row in enumerate(rows):
                writer.writerow([index] + row)
        return path

    def test_large_csv_stream_preserves_target_history_and_extrema(self):
        path = self._field('hs_length', [[1.0, 2.0, 0.0],
                                         [3.0, -99.0, 1.0],
                                         [5.0, 6.0, 2.0]])
        result = stream_field(path, [20], [(0.0, 2.0)], [])
        self.assertEqual(result['target_histories']['20'],
                         [[0.0, 2.0], [1.0, -99.0], [2.0, 6.0]])
        self.assertEqual(result['minimum']['global_quadrature_dof'], 20)
        self.assertEqual(result['minimum']['time'], 1.0)

    def test_report_records_missing_ecm_fields_and_verified_mapping(self):
        for field, values in (('lx', [1.0, 2.0]), ('ly', [3.0, 4.0]),
                              ('lz', [5.0, 6.0]), ('hs_length', [7.0, 8.0])):
            self._field(field, [values + [0.0], values + [1.0],
                                values + [2.0]])
        output = os.path.join(self.root, 'report')
        status = main([self.root, '--output', output, '--dofs', '10', '20',
                       '--fields', 'lx', 'ly', 'lz', 'hs_length',
                       'ecm_instantaneous_fiber_stress', '--no-plots'])
        self.assertEqual(status, 0)
        report = json.load(open(os.path.join(
            output, 'ecm_spatial_anomaly_report.json')))
        self.assertTrue(report['mapping']['verified'])
        self.assertIn('ecm_instantaneous_fiber_stress',
                      report['missing_fields'])


if __name__ == '__main__':
    unittest.main()
