import math
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORMS = ROOT / "python_codes/LV_simulation/dependencies/forms.py"
CONFIG = ROOT / "demos/base/sim_inputs/base_instruction.json"


PARAMETERS = {
    "myofiber": {"c2": 2.0, "c3": 3.0},
    "ground_matrix": {"a_g": 4.0, "b_g": 0.5},
    "collagen": {
        "a_cf": 5.0, "b_cf": 1.5,
        "a_cs": 6.0, "b_cs": 2.0,
        "a_cn": 7.0, "b_cn": 2.5,
    },
}


def xi_energy(stretch, parameters=PARAMETERS):
    values = parameters["myofiber"]
    xi = values["c3"]*max(stretch - 1.0, 0.0)**2
    return values["c2"]*(math.exp(xi) - 1.0)


def ground_energy(I1, parameters=PARAMETERS):
    values = parameters["ground_matrix"]
    return values["a_g"]/(2.0*values["b_g"])*(
        math.exp(values["b_g"]*(I1 - 3.0)) - 1.0)


def collagen_direction_energy(I4, direction, parameters=PARAMETERS):
    values = parameters["collagen"]
    I4_star = max(I4, 1.0)
    a = values["a_c" + direction]
    b = values["b_c" + direction]
    return a/(2.0*b)*(math.exp(b*(I4_star - 1.0)**2) - 1.0)


class HybridPassiveLawTests(unittest.TestCase):
    def test_reference_state_is_zero(self):
        self.assertEqual(ground_energy(3.0), 0.0)
        self.assertEqual(xi_energy(1.0), 0.0)
        for direction in ("f", "s", "n"):
            self.assertEqual(collagen_direction_energy(1.0, direction), 0.0)

    def test_collagen_is_inactive_in_compression(self):
        for direction in ("f", "s", "n"):
            self.assertEqual(collagen_direction_energy(0.8, direction), 0.0)
            step = 1.0e-7
            derivative = (
                collagen_direction_energy(1.0, direction) -
                collagen_direction_energy(1.0-step, direction))/step
            self.assertEqual(derivative, 0.0)

    def test_collagen_is_positive_and_exponential_in_tension(self):
        for direction in ("f", "s", "n"):
            low = collagen_direction_energy(1.1, direction)
            high = collagen_direction_energy(1.2, direction)
            self.assertGreater(low, 0.0)
            self.assertGreater(high, 2.0*low)

    def test_ground_matrix_has_requested_I1_law(self):
        I1 = 3.2
        expected = 4.0/(2.0*0.5)*(math.exp(0.5*(I1-3.0))-1.0)
        self.assertAlmostEqual(ground_energy(I1), expected)

    def test_reference_parameters_are_converted_from_kpa_to_pa(self):
        data = json.loads(CONFIG.read_text())
        values = data["mesh"]["forms_parameters"]["passive_law_parameters"]
        self.assertEqual(values["ground_matrix"]["a_g"][0], 352.7)
        self.assertEqual(values["collagen"]["a_cf"][0], 29772.0)
        self.assertEqual(values["collagen"]["a_cs"][0], 54744.0)
        self.assertEqual(values["collagen"]["a_cn"][0], 54744.0)

    def test_constituent_parameter_sensitivity_is_isolated(self):
        baseline = (
            ground_energy(3.2), xi_energy(1.1),
            collagen_direction_energy(1.1, "f"),
            collagen_direction_energy(1.1, "s"),
            collagen_direction_energy(1.1, "n"))
        paths = (
            ("ground_matrix", "a_g", 0),
            ("ground_matrix", "b_g", 0),
            ("myofiber", "c2", 1),
            ("myofiber", "c3", 1),
            ("collagen", "a_cf", 2),
            ("collagen", "b_cf", 2),
            ("collagen", "a_cs", 3),
            ("collagen", "b_cs", 3),
            ("collagen", "a_cn", 4),
            ("collagen", "b_cn", 4))
        for group, name, changed_index in paths:
            changed = {key: dict(value) for key, value in PARAMETERS.items()}
            changed[group][name] *= 2.0
            response = (
                ground_energy(3.2, changed), xi_energy(1.1, changed),
                collagen_direction_energy(1.1, "f", changed),
                collagen_direction_energy(1.1, "s", changed),
                collagen_direction_energy(1.1, "n", changed))
            for index, (old, new) in enumerate(zip(baseline, response)):
                if index == changed_index:
                    self.assertNotEqual(old, new)
                else:
                    self.assertEqual(old, new)

    def test_ufl_tension_only_conditionals_cover_all_directions(self):
        source = FORMS.read_text()
        for name in ("I4cf", "I4cs", "I4cn"):
            self.assertIn(
                "%s_star = conditional(%s > 1.0, %s, 1.0)" %
                (name, name, name), source)


if __name__ == "__main__":
    unittest.main()
