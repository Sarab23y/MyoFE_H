import csv
import json
import os
import shutil
import tempfile
import unittest

from validation.analyze_ecm_spatial_anomalies import (
    MATPLOTLIB_UNAVAILABLE_MESSAGE, _windows_for_time, main, stream_field)


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

    def test_overlapping_windows_each_receive_matching_samples(self):
        path = self._field('hs_length', [[1.0, 2.0, 38.0],
                                         [3.0, 4.0, 39.5],
                                         [5.0, 6.0, 40.0]])
        windows = [(35.0, 40.0), (39.0, 40.0)]
        result = stream_field(path, [], windows, [])
        broad, nested = result['window_summaries']
        self.assertEqual(broad['count'], 6)
        self.assertEqual(broad['mean'], 3.5)
        self.assertEqual(nested['count'], 4)
        self.assertEqual(nested['mean'], 4.5)
        self.assertEqual(_windows_for_time(39.5, windows), [0, 1])

    def test_non_overlapping_windows_receive_only_their_samples(self):
        path = self._field('hs_length', [[1.0, 3.0, 1.0],
                                         [5.0, 7.0, 11.0],
                                         [9.0, 11.0, 30.0]])
        windows = [(0.0, 2.0), (10.0, 12.0)]
        result = stream_field(path, [], windows, [])
        first, second = result['window_summaries']
        self.assertEqual((first['count'], first['mean']), (2, 2.0))
        self.assertEqual((second['count'], second['mean']), (2, 6.0))
        self.assertEqual(_windows_for_time(30.0, windows), [])

    def test_missing_matplotlib_message_explains_png_remediation(self):
        self.assertIn('PNG files were not generated',
                      MATPLOTLIB_UNAVAILABLE_MESSAGE)
        self.assertIn('Install matplotlib', MATPLOTLIB_UNAVAILABLE_MESSAGE)
        self.assertIn('without --no-plots', MATPLOTLIB_UNAVAILABLE_MESSAGE)

    def test_report_warns_about_historical_passive_field_ordering(self):
        self._field('bulk_passive', [[1.0, 2.0, 0.0]])
        output = os.path.join(self.root, 'warning-report')
        status = main([self.root, '--output', output, '--dofs', '10',
                       '--fields', 'bulk_passive', '--no-plots'])
        self.assertEqual(status, 0)
        with open(os.path.join(
                output, 'ecm_spatial_anomaly_report.json')) as handle:
            report = json.load(handle)
        self.assertEqual(report['historical_ordering_warning_fields'],
                         ['bulk_passive'])
        self.assertTrue(any('DG1 values' in item
                            for item in report['limitations']))

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
        with open(os.path.join(
                output, 'ecm_spatial_anomaly_report.json')) as handle:
            report = json.load(handle)
        self.assertTrue(report['mapping']['verified'])
        self.assertIn('ecm_instantaneous_fiber_stress',
                      report['missing_fields'])


if __name__ == '__main__':
    unittest.main()
