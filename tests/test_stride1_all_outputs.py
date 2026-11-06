import json
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'demos', 'ECM', 'sim_input',
    'base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json')


class StrideOneAllOutputTest(unittest.TestCase):
    def setUp(self):
        with open(CONFIG) as handle:
            self.config = json.load(handle)

    def test_protocol_is_unperturbed_sixty_seconds(self):
        protocol = self.config['protocol']
        self.assertEqual(protocol['no_of_time_steps'], [60000])
        self.assertEqual(protocol['time_step'], [0.001])
        self.assertEqual(protocol['perturbation']['perturbations'], [])
        self.assertEqual(protocol['baroreflex']['activations'][0],
                         {'t_start_s': [5], 't_stop_s': [60]})

    def test_all_output_strides_are_one_and_buffered(self):
        output = self.config['output_handler']
        self.assertEqual(output['output_mode'], ['buffered'])
        self.assertEqual(output['save_outputs'], ['all'])
        for name in ('frequency_n', 'global_csv_frequency_n',
                     'spatial_csv_frequency_n', 'mesh_frequency_n'):
            self.assertEqual(output[name], [1])
        self.assertNotIn('spatial_csv_fields', output)

    def test_diagnostic_does_not_enable_remodelling(self):
        model = self.config['model']
        self.assertIn('ecm_stress_diagnostic', model)
        self.assertNotIn('growth', model)
        self.assertNotIn('fibrosis', model)
        self.assertNotIn('fiber_reorientation', model)
        self.assertEqual(model['ecm_stress_diagnostic']['activation_time'],
                         [40.0])

    def test_all_output_writer_restores_cellular_fields(self):
        path = os.path.join(ROOT, 'python_codes', 'LV_simulation',
                            'LV_simulation.py')
        with open(path) as handle:
            source = handle.read()
        self.assertIn("if self.save_outputs_mode == 'all':", source)
        self.assertIn('self.spatial_myof_data_fields +', source)
        self.assertIn('self.spatial_memb_data_fields +', source)
        self.assertIn('for f in self.spatial_memb_data_fields:', source)


if __name__ == '__main__':
    unittest.main()
