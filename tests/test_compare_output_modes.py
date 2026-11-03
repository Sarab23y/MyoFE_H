import csv
import os
import shutil
import tempfile
import unittest
try:
    import numpy as np
    from validation.compare_output_modes import main
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, 'NumPy is required by the CSV comparator')
class OutputModeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.legacy = os.path.join(self.root, 'legacy')
        self.buffered = os.path.join(self.root, 'buffered')
        os.makedirs(self.legacy); os.makedirs(self.buffered)

    def tearDown(self):
        shutil.rmtree(self.root)

    def _write(self, directory, filename, header, rows):
        with open(os.path.join(directory, filename), 'w', newline='') as handle:
            writer = csv.writer(handle); writer.writerow(header)
            writer.writerows(rows)

    def _complete_fixture(self):
        global_header = ['', 'time', 'pressure_ventricle',
                         'volume_ventricle', 'heart_rate']
        global_rows = [[0, 0.0, 10.0, 100.0, 70.0],
                       [1, .01, 11.0, 99.0, 70.1]]
        spatial_header = ['', '0', '1', 'time']
        spatial_rows = [[0, 1.0, 2.0, 0.0], [1, 1.1, 2.1, .01]]
        for directory in (self.legacy, self.buffered):
            self._write(directory, 'data.csv', global_header, global_rows)
            for field in ('active_stress', 'total_passive', 'Sff_mesh',
                          'hs_length', 'fiber_strain',
                          'ecm_instantaneous_fiber_stress'):
                self._write(directory, field + '_data.csv', spatial_header,
                            spatial_rows)
            np.save(os.path.join(directory, 'quadrature_dof.npy'),
                    np.asarray([[0., 0., 0.], [1., 0., 0.]]))
            np.save(os.path.join(directory, 'quadrature_global_dof.npy'),
                    np.asarray([0, 1]))

    def test_equal_outputs_pass(self):
        self._complete_fixture()
        report = os.path.join(self.root, 'report.json')
        self.assertEqual(main([self.legacy, self.buffered,
                               '--report', report]), 0)

    def test_missing_field_fails(self):
        self._complete_fixture()
        os.remove(os.path.join(self.buffered, 'Sff_mesh_data.csv'))
        report = os.path.join(self.root, 'report.json')
        self.assertEqual(main([self.legacy, self.buffered,
                               '--report', report]), 1)


if __name__ == '__main__':
    unittest.main()
