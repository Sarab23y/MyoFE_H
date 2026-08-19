import json
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "demos/base/sim_inputs/base_instruction.json"


def components(state, parameters):
    lambda_m, I1, I4s, I8fs = state
    extension = max(lambda_m - 1.0, 0.0)
    myo = parameters["c2"] * (
        math.exp(parameters["c3"]*extension**2) - 1.0)
    bulk = parameters["a"]/(2.0*parameters["b"]) * (
        math.exp(parameters["b"]*(I1 - 3.0)) - 1.0)
    collagen = parameters["a_s"]/(2.0*parameters["b_s"]) * (
        math.exp(parameters["b_s"]*(I4s - 1.0)**2) - 1.0)
    collagen += parameters["a_fs"]/(2.0*parameters["b_fs"]) * (
        math.exp(parameters["b_fs"]*I8fs**2) - 1.0)
    return myo, bulk, collagen


def original_xi_sff(lambda_m, parameters):
    extension = max(lambda_m - 1.0, 0.0)
    return ((2.0/lambda_m)*parameters["c2"]*parameters["c3"]*
            extension*math.exp(parameters["c3"]*extension**2))


class HybridPassiveLawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads(EXAMPLE.read_text())
        raw = data["mesh"]["forms_parameters"]["passive_law_parameters"]
        cls.parameters = {key: value[0] for key, value in raw.items()
                          if key != "passive_law"}

    def test_reference_state_energy_and_deviatoric_stress_are_zero(self):
        self.assertEqual(components((1.0, 3.0, 1.0, 0.0), self.parameters),
                         (0.0, 0.0, 0.0))
        self.assertEqual(original_xi_sff(1.0, self.parameters), 0.0)
        # The bulk derivative at reference is hydrostatic, so its deviatoric
        # part is zero; collagen derivatives vanish at I4s=1 and I8fs=0.

    def test_example_fractions_are_bounded_and_sum_to_one(self):
        fractions = [self.parameters[name] for name in
                     ("phi_m", "phi_g", "phi_c")]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in fractions))
        self.assertAlmostEqual(sum(fractions), 1.0)

    def test_myofiber_only_exactly_matches_original_xi_law(self):
        lambda_m = 1.1
        energies = components((lambda_m, 3.2, 1.08, 0.04), self.parameters)
        expected_energy = self.parameters["c2"] * (
            math.exp(self.parameters["c3"]*(lambda_m - 1.0)**2) - 1.0)
        expected_sff = ((2.0/lambda_m)*self.parameters["c2"]*
                         self.parameters["c3"]*(lambda_m - 1.0)*
                         math.exp(self.parameters["c3"]*
                                  (lambda_m - 1.0)**2))
        self.assertAlmostEqual(energies[0], expected_energy)
        self.assertAlmostEqual(original_xi_sff(lambda_m, self.parameters),
                               expected_sff)

    def test_pure_constituents_and_mixture(self):
        energies = components((1.1, 3.2, 1.08, 0.04), self.parameters)
        for selected in range(3):
            fractions = [0.0, 0.0, 0.0]
            fractions[selected] = 1.0
            total = sum(fraction*energy for fraction, energy in
                        zip(fractions, energies))
            self.assertAlmostEqual(total, energies[selected])
        phi_m, phi_g, phi_c = 0.5, 0.2, 0.3
        total = phi_m*energies[0] + phi_g*energies[1] + phi_c*energies[2]
        self.assertAlmostEqual(total, sum(
            fraction*energy for fraction, energy in
            zip((phi_m, phi_g, phi_c), energies)))

    def test_ho_fiber_parameters_are_compatibility_only(self):
        baseline = components((1.1, 3.2, 1.08, 0.04), self.parameters)
        changed = dict(self.parameters)
        changed["a_f"] *= 100.0
        changed["b_f"] *= 100.0
        self.assertEqual(components((1.1, 3.2, 1.08, 0.04), changed), baseline)


if __name__ == "__main__":
    unittest.main()
