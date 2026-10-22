import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ECMStressDiagnosticTests(unittest.TestCase):
    def test_pointwise_three_cycle_reference_and_excess(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
                                   'dependencies',
                                   'ecm_stress_diagnostic.py')).read()
        self.assertIn('len(cycle_fields) != 3', source)
        self.assertIn('np.mean(np.vstack(fields), axis=0)', source)
        self.assertIn('np.maximum(signed, 0.0)', source)

    def test_reference_requires_three_matching_finite_fields(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
                                   'dependencies',
                                   'ecm_stress_diagnostic.py')).read()
        self.assertIn('field.shape == fields[0].shape', source)
        self.assertIn('np.all(np.isfinite(result))', source)

    def test_single_case_protocol_and_fixed_composition(self):
        path = os.path.join(ROOT, 'demos', 'base', 'sim_inputs',
                            'base_instruction_Br_ECM_stress_diagnostic.json')
        case = json.load(open(path))
        self.assertNotIn('growth', case['model'])
        self.assertNotIn('fibrosis', case['model'])
        diag = case['model']['ecm_stress_diagnostic']
        self.assertFalse(diag['composition_updates_enabled'][0])
        self.assertEqual(diag['homeostatic_cycles'][0], 3)
        change = case['protocol']['perturbation']['perturbations'][0]
        self.assertEqual((change['t_start_s'][0], change['t_stop_s'][0],
                          change['total_change'][0]), (20, 30, 170))
        passive = case['mesh']['forms_parameters']['passive_law_parameters']
        self.assertEqual((passive['phi_m'][0], passive['phi_g'][0],
                          passive['phi_c'][0]), (0.700, 0.274, 0.026))

    def test_ecm_tensor_source_excludes_xi_active_and_pressure(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
                                   'fibrosis', 'ecm_stress.py')).read()
        expression = "f['ground_matrix_passive'] + f['collagen_passive']"
        self.assertIn(expression, source)
        self.assertIn("inner(f0, (" + expression + ")*f0)", source)


if __name__ == '__main__':
    unittest.main()
