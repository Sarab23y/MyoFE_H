import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python_codes'))
from LV_simulation.dependencies.ecm_stability import (stability_results,
    symmetric_relative_change)
from validation.diagnose_ecm_baseline_stability import (scalar_cycles,
    time_weighted_mean)


class ECMStressDiagnosticTests(unittest.TestCase):
    def test_symmetric_relative_change_handles_zero_explicitly(self):
        self.assertEqual(symmetric_relative_change(0.0, 0.0), 0.0)
        self.assertEqual(symmetric_relative_change(0.0, 2.0), 1.0)
        self.assertAlmostEqual(symmetric_relative_change(-2.0, -2.1), 0.1/2.1)

    def test_all_stability_criteria_are_reported(self):
        records = [dict(pressure=10.0, volume=100.0, heart_rate=60.0),
                   dict(pressure=10.1, volume=100.0, heart_rate=70.0),
                   dict(pressure=12.0, volume=100.0, heart_rate=70.0)]
        result = stability_results(records, [0.01, 0.08], 0.05)
        self.assertEqual(set(result[0]['changes']),
                         set(('pressure', 'volume', 'heart_rate', 'ecm_stress_field')))
        self.assertFalse(result[0]['passed']['heart_rate'])
        self.assertFalse(result[1]['passed']['pressure'])
        self.assertFalse(result[1]['passed']['ecm_stress_field'])

    def test_identical_cycles_pass(self):
        records = [dict(pressure=10.0, volume=100.0, heart_rate=60.0)]*3
        result = stability_results(records, [0.0, 0.0], 0.05)
        self.assertTrue(all(all(pair['passed'].values()) for pair in result))

    def test_time_weighted_mean_and_complete_cycle_selection(self):
        self.assertAlmostEqual(time_weighted_mean([0.0, 1.0, 3.0],
                                                  [0.0, 2.0, 2.0]), 5.0/3.0)
        rows = []
        for time in range(5):
            rows.append({'time': str(time), 'new_beat': '1' if time in (0, 2, 4) else '0',
                         'pressure_ventricle': str(time), 'volume_ventricle': '100',
                         'heart_rate': '60'})
        cycles = scalar_cycles(rows, 5.0, 2)
        self.assertEqual([(cycle['start'], cycle['stop']) for cycle in cycles],
                         [(0.0, 2.0), (2.0, 4.0)])
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

    def test_independent_40_second_baseline_case(self):
        path = os.path.join(ROOT, 'demos', 'ECM', 'sim_input',
                            'base_instruction_Br_ECM_40s.json')
        case = json.load(open(path))
        self.assertEqual(case['protocol']['time_step'][0], 0.001)
        self.assertEqual(case['protocol']['no_of_time_steps'][0], 80000)
        self.assertEqual(case['protocol']['baroreflex']['activations'][0],
                         {'t_start_s': [5], 't_stop_s': [80]})
        change = case['protocol']['perturbation']['perturbations']
        self.assertEqual(len(change), 1)
        self.assertEqual((change[0]['t_start_s'][0], change[0]['t_stop_s'][0],
                          change[0]['total_change'][0]), (40, 50, 170))
        diagnostic = case['model']['ecm_stress_diagnostic']
        self.assertEqual(diagnostic['activation_time'][0], 40.0)
        self.assertEqual(diagnostic['baseline_stability_relative_tolerance'][0], 0.05)
        self.assertFalse(diagnostic['composition_updates_enabled'][0])
        self.assertFalse(diagnostic['geometric_growth_enabled'][0])
        self.assertNotIn('growth', case['model'])
        self.assertNotIn('fibrosis', case['model'])
        passive = case['mesh']['forms_parameters']['passive_law_parameters']
        self.assertEqual(tuple(passive[name][0] for name in ('phi_m','phi_g','phi_c')),
                         (0.700, 0.274, 0.026))

    def test_40_second_slurm_email_and_single_run(self):
        path = os.path.join(ROOT, 'demos', 'ECM', 'run_br_ecm_40s.sh')
        source = open(path).read()
        self.assertIn('#SBATCH --mail-user=Sara.b@uky.edu', source)
        self.assertIn('#SBATCH --mail-type=END,FAIL', source)
        self.assertEqual(source.count('python MyoFE.py LV_sim'), 1)
        self.assertIn('--baseline-end 40 --ramp-end 50 --simulation-end 80', source)

    def test_ecm_tensor_source_excludes_xi_active_and_pressure(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
                                   'fibrosis', 'ecm_stress.py')).read()
        expression = "f['ground_matrix_passive'] + f['collagen_passive']"
        self.assertIn(expression, source)
        self.assertIn("inner(f0, (" + expression + ")*f0)", source)


if __name__ == '__main__':
    unittest.main()
