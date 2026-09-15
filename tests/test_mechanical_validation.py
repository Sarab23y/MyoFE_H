import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MechanicalValidationSourceTests(unittest.TestCase):
    def test_material_evaluator_reuses_production_forms(self):
        source = (ROOT / "validation/material_point.py").read_text()
        self.assertIn("class PrescribedCForms(Forms)", source)
        self.assertIn("forms._passive_energy_components", source)
        self.assertIn("forms._passive_stress_components", source)
        self.assertIn("constraint_pressure = material_cauchy[2, 2]", source)

    def test_inflation_reuses_pressure_residual_and_disables_active_stress(self):
        source = (ROOT / "validation/passive_lv_inflation.py").read_text()
        self.assertIn("params['Ftotal_gr'] + cavity_multiplier_pin", source)
        self.assertIn("cb_density.vector()[:] = 0.0", source)
        self.assertNotIn("run_simulation(", source)
        self.assertNotIn("implement_time_step(", source)

    def test_example_configs_are_separate_from_production_input(self):
        material = json.loads((ROOT /
            "validation/configs/material_validation.json").read_text())
        inflation = json.loads((ROOT /
            "validation/configs/passive_inflation.json").read_text())
        self.assertEqual(material['input_json'],
                         'demos/base/sim_inputs/base_instruction.json')
        self.assertEqual(inflation['input_json'],
                         'demos/base/sim_inputs/base_instruction.json')
        self.assertNotEqual(material['output_directory'],
                            inflation['output_directory'])
        self.assertGreater(inflation['minimum_pressure_increment_mmHg'], 0.0)

    def test_plotting_is_explicit(self):
        for filename in ('plot_material_validation.py',
                         'plot_passive_inflation.py'):
            source = (ROOT / 'postprocessing' / filename).read_text()
            self.assertIn("'--headless'", source)
            self.assertIn("'--show'", source)
            self.assertIn("'--save'", source)


if __name__ == '__main__':
    unittest.main()
