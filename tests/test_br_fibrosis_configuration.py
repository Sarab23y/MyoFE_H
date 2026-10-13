import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'demos/base/sim_inputs'


def read(name):
    with (CONFIG_DIR / name).open() as handle:
        return json.load(handle)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MASS = load_module('br_mass',
    'python_codes/LV_simulation/dependencies/constituent_mass.py')
EVOLUTION = load_module('br_fibrosis_evolution',
    'python_codes/LV_simulation/dependencies/fibrosis_evolution.py')


class BrFibrosisConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = read('base_instruction_Br.json')
        cls.control = read('base_instruction_Br_fibrosis_control.json')
        cls.stenosis = read('base_instruction_Br_fibrosis_AS.json')

    def passive(self, case):
        return case['mesh']['forms_parameters']['passive_law_parameters']

    def test_authoritative_initial_fractions_and_coefficients(self):
        passive = self.passive(self.base)
        self.assertEqual(passive['phi_m'], [0.700])
        self.assertEqual(passive['phi_g'], [0.274])
        self.assertEqual(passive['phi_c'], [0.026])
        self.assertEqual(passive['myofiber'], {'c2': [80.0], 'c3': [25.0]})
        self.assertEqual(sum(passive[key][0]
            for key in ('phi_m', 'phi_g', 'phi_c')), 1.0)

    def test_derived_cases_preserve_calibrated_sections(self):
        for case in (self.control, self.stenosis):
            self.assertEqual(self.passive(case), self.passive(self.base))
            self.assertEqual(case['model']['half_sarcomere'],
                             self.base['model']['half_sarcomere'])
            self.assertEqual(case['model']['circulation'],
                             self.base['model']['circulation'])
            self.assertEqual(case['model']['baroreflex'],
                             self.base['model']['baroreflex'])
            self.assertEqual(case['heart_rate'], self.base['heart_rate'])
            self.assertNotIn('growth', case['model'])

    def test_exact_protocol_and_as_perturbation(self):
        for case in (self.base, self.control, self.stenosis):
            self.assertEqual(case['protocol']['time_step'], [0.001])
            self.assertEqual(case['protocol']['no_of_time_steps'], [60000])
            self.assertEqual(case['protocol']['growth']['activations'][0],
                {'t_start_s': [20], 't_stop_s': [60]})
            self.assertEqual(case['protocol']['baroreflex']['activations'][0],
                {'t_start_s': [5], 't_stop_s': [60]})
        self.assertNotIn('perturbation', self.control['protocol'])
        perturbation = self.stenosis['protocol']['perturbation'][
            'perturbations'][0]
        self.assertEqual(perturbation, {
            'variable': ['aorta_resistance'], 'level': ['circulation'],
            't_start_s': [30], 't_stop_s': [40], 'total_change': [170]})

    def test_fibrosis_parameters_and_output_paths(self):
        for case in (self.control, self.stenosis):
            config = EVOLUTION.validate_fibrosis_config(
                case['model']['fibrosis'])
            self.assertEqual(config['tau_c'], 50.0)
            self.assertEqual(config['phi_c_max'], 0.15)
            self.assertEqual(config['remodeling_frequency_cycles'], 4)
        self.assertEqual(self.control['output_handler']['frequency_n'], [10])
        self.assertEqual(self.stenosis['output_handler']['frequency_n'], [10])
        self.assertEqual(self.control['output_handler']['output_data_path'], [
            '/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test_control/sim_output/data.csv'])
        self.assertEqual(self.stenosis['output_handler']['output_data_path'], [
            '/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test/sim_output/data.csv'])

    def test_authoritative_circulation_myo_and_baroreflex_values(self):
        circulation = self.base['model']['circulation']
        self.assertEqual(circulation['blood_volume'], [4.5])
        expected_resistance = {
            'aorta': 20, 'arteries': 20, 'arterioles': 787,
            'capillaries': 350, 'venules': 50, 'veins': 50,
            'ventricle': 10}
        for compartment in circulation['compartments']:
            self.assertEqual(compartment['resistance'][0],
                             expected_resistance[compartment['name'][0]])
        myofilaments = self.base['model']['half_sarcomere']['myofilaments']
        for key, expected in {
                'k_1': 8, 'k_force': 0.001, 'k_2': 200, 'k_3': 80,
                'k_4_0': 80, 'k_4_1': 1.5, 'k_cb': 0.001,
                'x_ps': 5.2, 'k_on': 1.735e8, 'k_off': 200,
                'k_coop': 5}.items():
            self.assertEqual(myofilaments[key], [expected])
        baroreflex = self.base['model']['baroreflex']
        self.assertEqual(baroreflex['b_setpoint'], [90])
        self.assertEqual(baroreflex['b_slope'], [0.02])
        self.assertEqual(baroreflex['k_drive'], [20])
        self.assertEqual(baroreflex['k_recov'], [0.0])

    def test_mesh_path_and_solver(self):
        self.assertEqual(self.base['mesh']['mesh_path'], [
            '../demos/HCM_paper/simulations_baseline_final/'
            'baseline_final_Kappa_4_k1_3.7/sim_inputs/ellipsoidal.hdf5'])
        self.assertEqual(self.base['mesh']['solver']['params'], {
            'debugging_mode': [False], 'rel_tol': [1e-8],
            'abs_tol': [1e-8], 'max_iter': [100]})

    def test_constant_volume_exchange_uses_configured_fractions(self):
        EVOLUTION.validate_fraction_compatibility(
            [0.700], [0.274], [0.026], 0.15)
        with self.assertRaises(ValueError):
            EVOLUTION.validate_fraction_compatibility(
                [0.700], [0.274], [0.026], 0.301)
        bookkeeper = MASS.ConstituentMassBookkeeper(
            {'phi_m': [0.700], 'phi_g': [0.274], 'phi_c': [0.026]}, 1,
            {'enabled': [True], 'density': [1.0],
             'positive_growth_allocation': {'mode': ['proportional']}})
        before = dict((name, values[0]) for name, values in
                      bookkeeper.referential_density.items())
        state = bookkeeper.exchange_ground_matrix_for_collagen([0.15])
        after = dict((name, values[0]) for name, values in
                     state['referential_density'].items())
        self.assertEqual(after['myofiber'], before['myofiber'])
        self.assertAlmostEqual(after['collagen'] - before['collagen'], 0.124)
        self.assertAlmostEqual(before['ground_matrix'] - after['ground_matrix'],
                               0.124)
        self.assertAlmostEqual(sum(after.values()), sum(before.values()))
        self.assertAlmostEqual(state['fractions']['phi_g'][0], 0.15)

    def test_requested_growth_only_outputs_are_safely_excluded(self):
        unsupported = {'local_theta_vis_sheet', 'global_theta_vis_sheet',
            'local_theta_vis_fiber', 'global_theta_vis_fiber',
            'deviation_sheet', 'stimulus_sheet', 'setpoint_sheet',
            'local_theta_vis_sheet_normal', 'global_theta_vis_sheet_normal',
            'deviation_sheet_normal', 'stimulus_sheet_normal',
            'setpoint_sheet_normal', 'deviation_fiber', 'stimulus_fiber',
            'setpoint_fiber'}
        for case in (self.control, self.stenosis):
            outputs = set(case['output_handler']['mesh_object_to_save'])
            self.assertFalse(outputs.intersection(unsupported))
            self.assertIn('fibrosis_normalized_stimulus', outputs)
            self.assertIn('collagen_passive', outputs)

    def test_protocol_growth_window_is_consumed_without_geometric_growth(self):
        simulation = (ROOT /
            'python_codes/LV_simulation/LV_simulation.py').read_text()
        fibrosis = (ROOT /
            'python_codes/LV_simulation/fibrosis/fibrosis.py').read_text()
        self.assertIn(
            'self.fi.set_activation_window(self.prot.growth_activations)',
            simulation)
        self.assertIn('def set_activation_window', fibrosis)
        self.assertNotIn('ALE.move', fibrosis)


if __name__ == '__main__':
    unittest.main()
