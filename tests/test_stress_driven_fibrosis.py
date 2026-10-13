import importlib.util
import json
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
    def test_homeostatic_and_negative_stimulus_are_invariant(self):
        old = [0.03, 0.08]
        updated = EVOLUTION.exact_collagen_update(
            old, [0.0, -1.0], 4.0, 50.0, 0.15)
        self.assertEqual(updated, old)

    def test_exact_update_is_positive_bounded_and_saturates(self):
        old = [0.03]
        first = EVOLUTION.exact_collagen_update(
            old, [2.0], 4.0, 50.0, 0.15)
        self.assertGreater(first[0], old[0])
        self.assertLessEqual(first[0], 0.15)
        saturated = EVOLUTION.exact_collagen_update(
            old, [100.0], 100.0, 50.0, 0.15)
        self.assertAlmostEqual(saturated[0], 0.15)

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

    def test_configuration_rejects_geometric_growth_and_large_cap(self):
        config = read_json(
            ROOT / 'validation/configs/as_fibrosis_healthy.json')[
                'model']['fibrosis']
        validated = EVOLUTION.validate_fibrosis_config(config)
        self.assertFalse(config['geometric_growth_enabled'][0])
        self.assertTrue(validated['enabled'])
        bad = dict(config)
        bad['phi_c_max'] = [0.31]
        with self.assertRaises(ValueError):
            EVOLUTION.validate_fibrosis_config(bad)

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
