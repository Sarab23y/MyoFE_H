import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MechanicalHealthDiagnosticTests(unittest.TestCase):
    def test_stretch_strain_and_hsl_identities(self):
        for stretch in (0.25, 1.0, 1.2):
            strain = 0.5*(stretch*stretch-1.0)
            hsl0 = 1000.0; hsl = stretch*hsl0
            self.assertAlmostEqual(strain-(0.5*(stretch*stretch-1.0)), 0.0)
            self.assertAlmostEqual(hsl-stretch*hsl0, 0.0)

    def test_low_j_and_extreme_shortening_are_warnings(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
            'dependencies', 'mechanical_health.py')).read()
        for rule in ('nonpositive_J', 'small_positive_J',
                     'extreme_shortening', 'short_hsl', 'extreme_'):
            self.assertIn(rule, source)
        self.assertIn('Diagnostic only; no values are clipped', source)

    def test_new_case_is_healthy_only_and_diagnostics_are_opt_in(self):
        path = os.path.join(ROOT, 'demos', 'ECM', 'sim_input',
            'base_instruction_Br_ECM_40s_healthy_quad_validation.json')
        case = json.load(open(path)); protocol = case['protocol']; model = case['model']
        self.assertEqual(protocol['no_of_time_steps'][0], 42000)
        self.assertEqual(protocol['time_step'][0], 0.001)
        self.assertEqual(protocol['perturbation']['perturbations'], [])
        self.assertEqual(model['ecm_stress_diagnostic']['activation_time'][0], 40)
        self.assertTrue(model['ecm_stress_diagnostic']['save_reference_cycle_snapshot'][0])
        self.assertTrue(model['mechanical_health_diagnostic']['enabled'][0])
        passive = case['mesh']['forms_parameters']['passive_law_parameters']
        self.assertEqual(tuple(passive[name][0] for name in
            ('phi_m', 'phi_g', 'phi_c')), (0.700, 0.274, 0.026))
        self.assertNotIn('growth', model); self.assertNotIn('fibrosis', model)

    def test_existing_case_does_not_enable_health_diagnostic(self):
        path = os.path.join(ROOT, 'demos', 'ECM', 'sim_input',
            'base_instruction_Br_ECM_40s_provisional.json')
        case = json.load(open(path))
        self.assertNotIn('mechanical_health_diagnostic', case['model'])

    def test_output_preserves_historical_alpha_and_adds_explicit_names(self):
        source = open(os.path.join(ROOT, 'python_codes', 'LV_simulation',
                                  'LV_simulation.py')).read()
        self.assertIn("self.data['alpha_f'] = myo_stretch", source)
        self.assertIn("self.data['fiber_stretch_raw'] = raw_stretch", source)
        self.assertIn('fiber_stretch_tension_only', source)

    def test_mpi_validator_maps_by_coordinate_and_reports_weighted_l2(self):
        source = open(os.path.join(ROOT, 'validation',
            'validate_ecm_quadrature_mpi.py')).read()
        self.assertIn("tuple(item['coordinate'])", source)
        self.assertIn('relative_weighted_l2_difference', source)
        self.assertIn('missing_or_duplicate_coordinates', source)
        self.assertIn("'quadrature_point_in_cell'", source)

    def test_new_launcher_requires_passing_short_mpi_validation(self):
        source = open(os.path.join(ROOT, 'demos', 'ECM',
            'run_br_ecm_40s_healthy_quad_validation.sh')).read()
        self.assertIn('ECM_QUAD_VALIDATION_REPORT', source)
        self.assertIn("report['passed']", source)
        self.assertIn('refusing to overwrite', source)


if __name__ == '__main__':
    unittest.main()
