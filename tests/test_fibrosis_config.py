import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "demos/base/sim_inputs/base_instruction.json"
MODULE_PATH = (
    ROOT / "python_codes/LV_simulation/dependencies/fibrosis_config.py")
FORMS_PATH = ROOT / "python_codes/LV_simulation/dependencies/forms.py"
SPEC = importlib.util.spec_from_file_location("fibrosis_config", MODULE_PATH)
FIBROSIS_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIBROSIS_CONFIG)


def passive_parameters():
    data = json.loads(CONFIG_PATH.read_text())
    return data["mesh"]["forms_parameters"]["passive_law_parameters"]


class FibrosisConfigurationTests(unittest.TestCase):
    def test_user_parameters_override_only_documented_form_defaults(self):
        source = FORMS_PATH.read_text()
        defaults = source.index("self.parameters = self.default_parameters()")
        user_update = source.index("self.parameters.update(params)")
        self.assertLess(defaults, user_update)

    def test_current_json_values_are_resolved_without_changes(self):
        parameters = passive_parameters()
        before = json.loads(json.dumps(parameters))
        resolved = FIBROSIS_CONFIG.validate_xi_ground_collagen(parameters)
        self.assertEqual(parameters, before)
        self.assertEqual(resolved["a_g"], 352.7)
        self.assertEqual(resolved["b_g"], 0.1158)
        self.assertEqual(resolved["c2"], 250)
        self.assertEqual(resolved["c3"], 15.0)
        self.assertEqual(resolved["a_cf"], 29772.0)
        self.assertEqual(resolved["b_cf"], 4.0985)
        self.assertEqual(resolved["a_cs"], 54744.0)
        self.assertEqual(resolved["b_cs"], 2.4095)
        self.assertEqual(resolved["a_cn"], 54744.0)
        self.assertEqual(resolved["b_cn"], 2.4095)
        self.assertEqual(resolved["phi_m"], 0.70)
        self.assertEqual(resolved["phi_g"], 0.27)
        self.assertEqual(resolved["phi_c"], 0.03)
        self.assertAlmostEqual(resolved["fraction_sum"], 1.0)

    def test_missing_parameter_reports_full_json_path(self):
        parameters = passive_parameters()
        del parameters["collagen"]["a_cf"]
        with self.assertRaisesRegex(
                ValueError,
                r"passive_law_parameters\.collagen\.a_cf"):
            FIBROSIS_CONFIG.validate_xi_ground_collagen(parameters)

    def test_fraction_error_does_not_normalize_input(self):
        parameters = passive_parameters()
        parameters["phi_c"][0] = 0.04
        before = json.loads(json.dumps(parameters))
        with self.assertRaisesRegex(ValueError, r"phi_m \+ phi_g \+ phi_c"):
            FIBROSIS_CONFIG.validate_xi_ground_collagen(parameters)
        self.assertEqual(parameters, before)


if __name__ == "__main__":
    unittest.main()
