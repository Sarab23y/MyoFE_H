import csv
import os
import tempfile
import unittest

from python_codes.LV_simulation.dependencies.buffered_output import (
    BoundedRowBuffer, estimated_numeric_buffer_bytes, is_scalar_output_value,
    scalar_field_names,
    validate_global_dof_partition)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BufferedOutputTests(unittest.TestCase):
    def test_scalar_spatial_classification(self):
        class ZeroDimensionalNumeric(object):
            ndim = 0
            dtype = type('DType', (), {'kind': 'f'})()
        self.assertTrue(is_scalar_output_value(1.0))
        self.assertTrue(is_scalar_output_value(ZeroDimensionalNumeric()))
        self.assertFalse(is_scalar_output_value([1.0]))
        self.assertEqual(scalar_field_names(
            {'pressure': 10.0, 'fiber_stretch_raw': [1.0]*4}),
            ['pressure'])

    def test_global_writer_rejects_shape_change(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
                                  'LV_simulation.py')).read()
        self.assertIn('Global output field %s changed from a', source)
        self.assertIn('if is_scalar_output_value(value)', source)

    def test_fenics_2017_diagnostic_reuses_evaluated_stretch(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
                                  'LV_simulation.py')).read()
        block = source[source.index('# hsl and hsl0 are stored'):
                       source.index('if self.fi:', source.index(
                           '# hsl and hsl0 are stored'))]
        self.assertIn('new_hs_length_list', block)
        self.assertNotIn('project(', block)
        health = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
            'dependencies', 'mechanical_health.py')).read()
        self.assertIn("self.parent.data['fiber_stretch_raw']", health)
        self.assertNotIn("'representation': 'quadrature'", health)

    def _buffered(self, count, frequency, batch_size):
        buffer = BoundedRowBuffer(('time', 'pressure', 'volume'), batch_size)
        written = []
        for step in range(count+1):
            if step % frequency:
                continue
            full = buffer.append({'time': 0.001*step,
                                  'pressure': 2.0*step,
                                  'volume': 100.0-step})
            if full:
                written.extend(buffer.drain())
        written.extend(buffer.drain())
        return written

    def test_frequency_one_matches_legacy_rows(self):
        expected = [(0.001*i, 2.0*i, 100.0-i) for i in range(11)]
        self.assertEqual(self._buffered(10, 1, 4), expected)

    def test_frequency_ten_and_partial_final_batch(self):
        rows = self._buffered(25, 10, 2)
        self.assertEqual([row[0] for row in rows], [0.0, 0.01, 0.02])

    def test_repeated_empty_drain_never_duplicates_rows(self):
        buffer = BoundedRowBuffer(('time',), 2)
        buffer.append({'time': 0.0})
        self.assertEqual(buffer.drain(), [(0.0,)])
        self.assertEqual(buffer.drain(), [])

    def test_early_failure_preserves_only_completed_rows(self):
        buffer = BoundedRowBuffer(('time',), 100)
        for value in (0.0, 0.1, 0.2):
            buffer.append({'time': value})
        self.assertEqual(buffer.drain(), [(0.0,), (0.1,), (0.2,)])

    def test_global_dof_partition_validation(self):
        self.assertTrue(validate_global_dof_partition([[0, 2], [1, 3]], 4))
        self.assertFalse(validate_global_dof_partition([[0, 1], [1, 3]], 4))
        self.assertFalse(validate_global_dof_partition([[0, 1], [2]], 4))

    def test_memory_estimate_is_independent_of_duration(self):
        bounded = estimated_numeric_buffer_bytes(5, 2500, 100, 20)
        legacy = estimated_numeric_buffer_bytes(5, 2500, 8001, 20)
        self.assertLess(bounded, legacy/79.0)

    def test_physical_feedback_paths_are_not_conditioned_on_output(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
                                  'LV_simulation.py')).read()
        for token in ('self.hs_objs_list[j].update_simulation',
                      'self.solver.solvenonlinear()',
                      'self.fi.record_time_step',
                      'self.ecm_diag.record_time_step'):
            self.assertIn(token, source)
        self.assertIn('Output-only projections are evaluated lazily', source)
        self.assertIn('self.f0_vs_time_array = None', source)
        self.assertIn('open_memmap', source)

    def test_optimized_case_preserves_analysis_contract(self):
        import json
        path = os.path.join(ROOT, 'demos', 'ECM', 'sim_input',
            'base_instruction_Br_ECM_40s_healthy_quad_validation_optimized.json')
        case = json.load(open(path)); output = case['output_handler']
        self.assertEqual(output['output_mode'][0], 'buffered')
        self.assertEqual(output['output_batch_size'][0], 100)
        self.assertIn('ecm_instantaneous_fiber_stress',
                      output['spatial_csv_fields'])
        self.assertIn('fiber_stretch_raw', output['spatial_csv_fields'])
        self.assertEqual(case['protocol']['perturbation']['perturbations'], [])

    def test_smoke_cases_have_identical_physics(self):
        import json
        root = os.path.join(ROOT, 'demos', 'ECM', 'sim_input')
        cases = [json.load(open(os.path.join(root, name))) for name in (
            'base_instruction_ECM_output_smoke_legacy.json',
            'base_instruction_ECM_output_smoke_buffered.json')]
        outputs = [case.pop('output_handler') for case in cases]
        self.assertEqual(cases[0], cases[1])
        self.assertEqual(cases[0]['protocol']['no_of_time_steps'][0], 200)
        self.assertEqual(cases[0]['protocol']['time_step'][0], 0.001)
        self.assertNotIn('output_mode', outputs[0])
        self.assertEqual(outputs[1]['output_mode'][0], 'buffered')


if __name__ == '__main__':
    unittest.main()
