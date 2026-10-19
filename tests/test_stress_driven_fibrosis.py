import importlib.util
import json
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVOLUTION = load('fibrosis_evolution',
                 'python_codes/LV_simulation/dependencies/fibrosis_evolution.py')
MASS = load('constituent_mass_fibrosis',
            'python_codes/LV_simulation/dependencies/constituent_mass.py')


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


class StressDrivenFibrosisTests(unittest.TestCase):
    def test_gamma_one_reproduces_previous_update(self):
        previous = 0.026 + (0.15 - 0.026)*(
            1.0 - math.exp(-0.75*4.0/50.0))
        explicit = EVOLUTION.exact_collagen_update(
            [0.026], [0.75], 4.0, 50.0, 0.15, gamma_c=1.0)
        self.assertAlmostEqual(previous, explicit[0])

    def test_nonlinear_effective_stimulus(self):
        effective = EVOLUTION.effective_fibrosis_stimulus(
            [0.0, 0.5, 1.0, 2.0], 1.5)
        self.assertEqual(effective[0], 0.0)
        self.assertLess(effective[1], 0.5)
        self.assertEqual(effective[2], 1.0)
        self.assertGreater(effective[3], 2.0)

    def test_gamma_is_applied_once_after_positive_part(self):
        evolution = (ROOT /
            'python_codes/LV_simulation/dependencies/fibrosis_evolution.py'
            ).read_text()
        self.assertEqual(evolution.count('raw_value**gamma_c'), 1)
        self.assertIn('raw_value = max(float(raw_value), 0.0)', evolution)
        fibrosis = (ROOT /
            'python_codes/LV_simulation/fibrosis/fibrosis.py').read_text()
        self.assertIn(
            "fractions['phi_c'], stimulus, self._remodeling_duration,",
            fibrosis)
        self.assertNotIn(
            "fractions['phi_c'], effective_stimulus,", fibrosis)

    def test_homeostatic_and_negative_stimulus_are_invariant(self):
        old = [0.03, 0.08]
        updated = EVOLUTION.exact_collagen_update(
            old, [0.0, -1.0], 4.0, 50.0, 0.15)
        self.assertEqual(updated, old)
        nonlinear = EVOLUTION.exact_collagen_update(
            old, [0.0, 0.0], 4.0, 50.0, 0.15, gamma_c=1.5)
        self.assertEqual(nonlinear, old)

    def test_exact_update_is_positive_bounded_and_saturates(self):
        old = [0.03]
        first = EVOLUTION.exact_collagen_update(
            old, [2.0], 4.0, 50.0, 0.15)
        self.assertGreater(first[0], old[0])
        self.assertLessEqual(first[0], 0.15)
        saturated = EVOLUTION.exact_collagen_update(
            old, [100.0], 100.0, 50.0, 0.15)
        self.assertAlmostEqual(saturated[0], 0.15)


    def test_signed_range_signal_at_bounds_and_midpoint(self):
        normalized, signal = EVOLUTION.stress_range_signal(
            [-10.0, 0.0, 10.0], -10.0, 10.0, 1.5, 50.0)
        self.assertEqual(normalized, [0.0, 0.5, 1.0])
        self.assertEqual(signal[0], 0.0)
        self.assertAlmostEqual(signal[1], 0.5**1.5/50.0)
        self.assertAlmostEqual(signal[2], 1.0/50.0)

    def test_signed_range_signal_handles_negative_stress_and_clips(self):
        normalized, signal = EVOLUTION.stress_range_signal(
            [-20.0, -5.0, 20.0], -10.0, 10.0, 1.5, 50.0)
        self.assertEqual(normalized, [0.0, 0.25, 1.0])
        self.assertEqual(signal[0], 0.0)
        self.assertAlmostEqual(signal[1], 0.25**1.5/50.0)
        self.assertAlmostEqual(signal[2], 1.0/50.0)

    def test_range_update_applies_tau_and_gamma_exactly_once(self):
        normalized, signal = EVOLUTION.stress_range_signal(
            [0.0], -10.0, 10.0, 1.5, 50.0)
        updated = EVOLUTION.exact_collagen_update_from_signal(
            [0.026], signal, 4.0, 0.15)
        expected = 0.026 + (0.15 - 0.026)*(1.0 - math.exp(
            -(0.5**1.5/50.0)*4.0))
        self.assertAlmostEqual(updated[0], expected)
        self.assertEqual(normalized, [0.5])

    def test_range_configuration_requires_valid_fixed_limits(self):
        config = read_json(
            ROOT / 'validation/configs/as_fibrosis_healthy.json')[
                'model']['fibrosis']
        config['stimulus'] = ['cycle_averaged_total_fiber_stress_range']
        for bad_limits in ((None, None), (1.0, 1.0), (2.0, 1.0),
                           (float('nan'), 1.0), (-1.0, float('inf'))):
            bad = dict(config)
            if bad_limits[0] is not None:
                bad['S_ff_min'] = [bad_limits[0]]
                bad['S_ff_max'] = [bad_limits[1]]
            with self.assertRaises(ValueError):
                EVOLUTION.validate_fibrosis_config(bad)
        config['S_ff_min'] = [-10.0]
        config['S_ff_max'] = [10.0]
        validated = EVOLUTION.validate_fibrosis_config(config)
        self.assertEqual(validated['S_ff_min'], -10.0)
        self.assertEqual(validated['S_ff_max'], 10.0)


    def test_range_case_generator_preserves_sources_and_sets_outputs(self):
        generator = load('prepare_stress_range_cases',
                         'validation/prepare_stress_range_cases.py')
        source_path = ROOT / ('demos/base/sim_inputs/'
                              'base_instruction_Br_fibrosis_control.json')
        original = source_path.read_bytes()
        case = generator.configure(read_json(source_path), -10.0, 10.0,
                                   '/tmp/new-range-control')
        self.assertEqual(source_path.read_bytes(), original)
        fibrosis = case['model']['fibrosis']
        self.assertEqual(fibrosis['stimulus'],
                         ['cycle_averaged_total_fiber_stress_range'])
        self.assertEqual(fibrosis['S_ff_min'], [-10.0])
        self.assertEqual(fibrosis['S_ff_max'], [10.0])
        self.assertNotIn('growth', case['model'])
        self.assertEqual(case['output_handler']['output_data_path'],
                         ['/tmp/new-range-control/sim_output/data.csv'])

    def test_exchange_preserves_myofiber_and_total_density(self):
        bookkeeper = MASS.ConstituentMassBookkeeper(
            {'phi_m': [0.70], 'phi_g': [0.27], 'phi_c': [0.03]}, 1,
            {'enabled': [True], 'density': [1.05],
             'positive_growth_allocation': {'mode': ['proportional']}})
        before_m = bookkeeper.referential_density['myofiber'][0]
        before_total = sum(v[0] for v in bookkeeper.referential_density.values())
        state = bookkeeper.exchange_ground_matrix_for_collagen([0.12])
        after_total = sum(v[0] for v in state['referential_density'].values())
        self.assertAlmostEqual(
            state['referential_density']['myofiber'][0], before_m)
        self.assertAlmostEqual(after_total, before_total)
        self.assertAlmostEqual(state['fractions']['phi_m'][0], 0.70)
        self.assertAlmostEqual(state['fractions']['phi_g'][0], 0.18)
        self.assertAlmostEqual(state['fractions']['phi_c'][0], 0.12)
        self.assertAlmostEqual(sum(state['fractions'][key][0]
            for key in ('phi_m', 'phi_g', 'phi_c')), 1.0)

    def test_configuration_rejects_geometric_growth_and_invalid_cap(self):
        config = read_json(
            ROOT / 'validation/configs/as_fibrosis_healthy.json')[
                'model']['fibrosis']
        validated = EVOLUTION.validate_fibrosis_config(config)
        self.assertFalse(config['geometric_growth_enabled'][0])
        self.assertTrue(validated['enabled'])
        bad = dict(config)
        bad['phi_c_max'] = [1.0]
        with self.assertRaises(ValueError):
            EVOLUTION.validate_fibrosis_config(bad)
        for invalid in (0.0, -1.0, float('inf'), float('nan')):
            bad = dict(config)
            bad['gamma_c'] = [invalid]
            with self.assertRaises(ValueError):
                EVOLUTION.validate_fibrosis_config(bad)

    def test_absent_gamma_defaults_to_linear_model(self):
        config = read_json(
            ROOT / 'validation/configs/as_fibrosis_healthy.json')[
                'model']['fibrosis']
        config.pop('gamma_c', None)
        self.assertEqual(
            EVOLUTION.validate_fibrosis_config(config)['gamma_c'], 1.0)

    def test_as_case_reuses_existing_resistance_perturbation(self):
        original = read_json(
            ROOT / 'demos/growth_AS/sim_inputs/input_parameters.json')
        candidate = read_json(
            ROOT / 'validation/configs/as_fibrosis_aortic_stenosis.json')
        original_as = original['protocol']['perturbation']['perturbations'][0]
        candidate_as = candidate['protocol']['perturbation']['perturbations'][0]
        self.assertEqual(candidate_as, original_as)

    def test_cases_preserve_current_baseline_science(self):
        baseline = read_json(
            ROOT / 'demos/base/sim_inputs/base_instruction.json')
        for filename in ('as_fibrosis_healthy.json',
                         'as_fibrosis_aortic_stenosis.json'):
            case = read_json(ROOT / 'validation/configs' / filename)
            self.assertEqual(case['mesh']['forms_parameters'],
                             baseline['mesh']['forms_parameters'])
            self.assertEqual(case['model']['half_sarcomere'],
                             baseline['model']['half_sarcomere'])
            self.assertEqual(case['model']['circulation'],
                             baseline['model']['circulation'])
            self.assertNotIn('growth', case['model'])

    def test_fibrosis_path_cannot_move_reference_mesh(self):
        source = (ROOT /
            'python_codes/LV_simulation/fibrosis/fibrosis.py').read_text()
        self.assertNotIn('ALE.move', source)
        self.assertIn("if 'growth' in parent.instruction_data['model']", source)
        self.assertIn('record_time_step(time_step, self.end_diastolic)',
            (ROOT / 'python_codes/LV_simulation/LV_simulation.py').read_text())

    def test_existing_weighted_passive_energy_is_unchanged_and_coupled(self):
        forms = (ROOT /
            'python_codes/LV_simulation/dependencies/forms.py').read_text()
        self.assertIn('phi_g*W_ground + phi_m*W_myo + phi_c*W_collagen',
                      forms)
        fibrosis = (ROOT /
            'python_codes/LV_simulation/fibrosis/fibrosis.py').read_text()
        self.assertIn("passive[key][-1]", fibrosis)
        self.assertIn("function.vector().apply('insert')", fibrosis)


if __name__ == '__main__':
    unittest.main()
