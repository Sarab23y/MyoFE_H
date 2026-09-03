import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORMS = ROOT / "python_codes/LV_simulation/dependencies/forms.py"


class ConstitutiveImplementationAudit(unittest.TestCase):
    def test_xi_energy_expression_is_preserved(self):
        source = FORMS.read_text()
        self.assertIn("myofiber_stretch = hsl/hsl0", source)
        self.assertIn('myofiber_parameters = self.parameters["myofiber"]', source)
        self.assertIn("C3*(myofiber_stretch - 1.0)**2.0", source)
        self.assertIn("W_myo = C2*(exp(Xi) - 1.0)", source)

    def test_xi_stress_expression_is_preserved(self):
        source = FORMS.read_text()
        self.assertIn("(2.0/myofiber_stretch)*C2*C3*", source)
        self.assertIn("myofiber_stretch > 1.0", source)
        self.assertIn("Sff = phi_m*Sff_unweighted", source)

    def test_constituent_energies_are_fraction_weighted(self):
        source = FORMS.read_text()
        self.assertIn(
            "return phi_g*W_ground + phi_m*W_myo + phi_c*W_collagen",
            source)

    def test_incompressibility_and_penalty_paths_remain(self):
        source = FORMS.read_text()
        self.assertIn("Wp - p*(self.J() - 1.0)", source)
        self.assertIn(
            'self.parameters["Kappa"]/2.0*(self.J() - 1.0)**2.0',
            source)


if __name__ == "__main__":
    unittest.main()
