import json
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "demos/base/sim_inputs/base_instruction.json"


def components(invariants, parameters):
    I1, I4f, I4s, I8fs = invariants
    I4f_eff = max(I4f, 1.0)
    I4s_eff = max(I4s, 1.0)
    bulk = parameters["a"]/(2.0*parameters["b"]) * (
        math.exp(parameters["b"]*(I1 - 3.0)) - 1.0)
    myo = parameters["a_f"]/(2.0*parameters["b_f"]) * (
        math.exp(parameters["b_f"]*(I4f_eff - 1.0)**2) - 1.0)
    collagen = parameters["a_s"]/(2.0*parameters["b_s"]) * (
        math.exp(parameters["b_s"]*(I4s_eff - 1.0)**2) - 1.0)
    collagen += parameters["a_fs"]/(2.0*parameters["b_fs"]) * (
        math.exp(parameters["b_fs"]*I8fs**2) - 1.0)
    return bulk, myo, collagen


class HolzapfelOgdenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads(EXAMPLE.read_text())
        raw = data["mesh"]["forms_parameters"]["passive_law_parameters"]
        cls.parameters = {key: value[0] for key, value in raw.items()
                          if key != "passive_law"}

    def test_undeformed_energy_is_zero(self):
        self.assertEqual(components((3.0, 1.0, 1.0, 0.0), self.parameters),
                         (0.0, 0.0, 0.0))

    def test_example_fractions_are_bounded_and_sum_to_one(self):
        fractions = [self.parameters[name] for name in
                     ("phi_m", "phi_c", "phi_g")]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in fractions))
        self.assertAlmostEqual(sum(fractions), 1.0)

    def test_each_pure_constituent_selects_only_its_energy(self):
        energies = components((3.2, 1.1, 1.08, 0.04), self.parameters)
        for selected in range(3):
            fractions = [0.0, 0.0, 0.0]
            fractions[selected] = 1.0
            total = sum(fraction*energy for fraction, energy in
                        zip(fractions, energies))
            self.assertAlmostEqual(total, energies[selected])

    def test_weighted_total_and_directional_stress_derivative(self):
        invariants = [3.2, 1.1, 1.08, 0.04]
        fractions = [0.2, 0.5, 0.3]  # bulk, myofiber, collagen
        energies = components(invariants, self.parameters)
        expected = sum(fraction*energy for fraction, energy in
                       zip(fractions, energies))
        self.assertAlmostEqual(expected,
            fractions[0]*energies[0] + fractions[1]*energies[1] +
            fractions[2]*energies[2])

        # Check the myofiber PK2 coefficient 2*dW/dI4f against a centered
        # finite difference of the exact same weighted energy.
        step = 1.0e-6
        def total_at(I4f):
            varied = list(invariants)
            varied[1] = I4f
            values = components(varied, self.parameters)
            return sum(fraction*energy for fraction, energy in
                       zip(fractions, values))
        finite_difference = (total_at(invariants[1] + step) -
                             total_at(invariants[1] - step))/step
        analytic = (2.0*fractions[1]*self.parameters["a_f"] *
                    (invariants[1] - 1.0) *
                    math.exp(self.parameters["b_f"] *
                             (invariants[1] - 1.0)**2))
        self.assertAlmostEqual(finite_difference, analytic, places=4)


if __name__ == "__main__":
    unittest.main()
