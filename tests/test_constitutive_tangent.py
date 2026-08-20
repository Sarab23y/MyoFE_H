import json
import math
import pathlib
import unittest



ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "demos/base/sim_inputs/base_instruction.json"
MODES = {
    "fiber": ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "sheet": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
    "sheet_normal": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "fs_shear": ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "fn_shear": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "sn_shear": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
}


def energies(F, parameters):
    C = [[sum(F[k][i]*F[k][j] for k in range(3))
          for j in range(3)] for i in range(3)]
    lambda_m = math.sqrt(C[0][0])
    I1 = sum(C[i][i] for i in range(3))
    I4s = C[1][1]
    I8fs = C[0][1]
    extension = max(lambda_m - 1.0, 0.0)
    W_myo = parameters["c2"]*(
        math.exp(parameters["c3"]*extension**2) - 1.0)
    W_bulk = parameters["a"]/(2.0*parameters["b"])*(
        math.exp(parameters["b"]*(I1 - 3.0)) - 1.0)
    collagen_extension = max(I4s - 1.0, 0.0)
    W_collagen = parameters["a_s"]/(2.0*parameters["b_s"])*(
        math.exp(parameters["b_s"]*collagen_extension**2) - 1.0)
    W_collagen += parameters["a_fs"]/(2.0*parameters["b_fs"])*(
        math.exp(parameters["b_fs"]*I8fs**2) - 1.0)
    return (W_myo, W_bulk, W_collagen)


def centered_curvature(mode, fractions, parameters, step=1.0e-5,
                       regularization=0.0):
    def weighted_energy(epsilon):
        F = tuple(tuple((1.0 if i == j else 0.0) + epsilon*mode[i][j]
                        for j in range(3)) for i in range(3))
        values = energies(F, parameters)
        return (sum(fraction*energy for fraction, energy in
                    zip(fractions, values)) + regularization*values[1])
    return ((weighted_energy(step) - 2.0*weighted_energy(0.0) +
             weighted_energy(-step))/step**2)


class ConstitutiveTangentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads(EXAMPLE.read_text())
        raw = data["mesh"]["forms_parameters"]["passive_law_parameters"]
        cls.parameters = {key: value[0] for key, value in raw.items()
                          if key != "passive_law"}

    def test_xi_one_sided_reference_tangent(self):
        # The strict UFL `lambda_m > 1` conditional selects zero stiffness at
        # exactly lambda_m=1. The compression-side tangent is zero, while the
        # tensile-side tangent is 2*c2*c3 and only supports fiber extension.
        self.assertEqual(0.0, 0.0)
        self.assertGreater(2.0*self.parameters["c2"]*self.parameters["c3"],
                           0.0)
        curvatures = {
            name: centered_curvature(mode, (1.0, 0.0, 0.0), self.parameters)
            for name, mode in MODES.items()}
        self.assertEqual(sum(value > 1.0e-6 for value in curvatures.values()),
                         1)
        self.assertGreater(curvatures["fiber"], 0.0)
        for name in ("sheet", "sheet_normal", "fs_shear", "fn_shear",
                     "sn_shear"):
            self.assertAlmostEqual(curvatures[name], 0.0, places=5)

    def test_collagen_reference_tangent_has_rank_two(self):
        curvatures = {
            name: centered_curvature(mode, (0.0, 0.0, 1.0), self.parameters)
            for name, mode in MODES.items()}
        self.assertEqual(sum(value > 1.0e-6 for value in curvatures.values()),
                         2)
        self.assertGreater(curvatures["sheet"], 0.0)
        self.assertGreater(curvatures["fs_shear"], 0.0)
        for name in ("fiber", "sheet_normal", "fn_shear", "sn_shear"):
            self.assertAlmostEqual(curvatures[name], 0.0, places=4)

    def test_debugging_mixtures_gain_stiffness_from_bulk(self):
        for fractions in ((0.95, 0.05, 0.0), (0.0, 0.05, 0.95)):
            curvatures = [centered_curvature(mode, fractions, self.parameters)
                          for mode in MODES.values()]
            self.assertTrue(all(value > 0.0 for value in curvatures))

    def test_requested_sensitivity_mixtures_have_all_mode_support(self):
        mixtures = (
            (0.10, 0.90, 0.00),
            (0.25, 0.75, 0.00),
            (0.00, 0.90, 0.10),
            (0.00, 0.75, 0.25),
            (0.20, 0.70, 0.10),
            (0.20, 0.60, 0.20))
        for fractions in mixtures:
            self.assertTrue(all(
                centered_curvature(mode, fractions, self.parameters) > 0.0
                for mode in MODES.values()))

    def test_regularization_is_zero_by_default_and_fraction_independent(self):
        self.assertEqual(self.parameters["passive_regularization"], 0.0)
        for fractions in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
            self.assertTrue(all(
                centered_curvature(
                    mode, fractions, self.parameters,
                    regularization=1.0e-6) > 0.0
                for mode in MODES.values()))


if __name__ == "__main__":
    unittest.main()
