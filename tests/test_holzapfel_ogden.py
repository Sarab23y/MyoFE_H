import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORMS = ROOT / "python_codes/LV_simulation/dependencies/forms.py"


PARAMETERS = {
    # Test-only values: production calibration remains configuration-owned.
    "c2": 2.0,
    "c3": 3.0,
    "a_g": 4.0,
    "b_g": 0.5,
    "a_cf": 5.0,
    "b_cf": 1.5,
    "a_cs": 6.0,
    "b_cs": 2.0,
    "a_cn": 7.0,
    "b_cn": 2.5,
}


def xi_energy(stretch, parameters=PARAMETERS):
    xi = parameters["c3"]*max(stretch - 1.0, 0.0)**2
    return parameters["c2"]*(math.exp(xi) - 1.0)


def ground_energy(I1, parameters=PARAMETERS):
    return parameters["a_g"]/(2.0*parameters["b_g"])*(
        math.exp(parameters["b_g"]*(I1 - 3.0)) - 1.0)


def collagen_direction_energy(I4, direction, parameters=PARAMETERS):
    I4_star = max(I4, 1.0)
    a = parameters["a_c" + direction]
    b = parameters["b_c" + direction]
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

    def test_ufl_tension_only_conditionals_cover_all_directions(self):
        source = FORMS.read_text()
        for name in ("I4cf", "I4cs", "I4cn"):
            self.assertIn(
                "%s_star = conditional(%s > 1.0, %s, 1.0)" %
                (name, name, name), source)


if __name__ == "__main__":
    unittest.main()
