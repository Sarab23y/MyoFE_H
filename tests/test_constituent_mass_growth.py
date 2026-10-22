import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / (
    "python_codes/LV_simulation/dependencies/constituent_mass.py")
SPEC = importlib.util.spec_from_file_location("constituent_mass", MODULE_PATH)
MASS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MASS)


def initial_fractions(points=2):
    return {
        "phi_m": [0.70]*points,
        "phi_g": [0.27]*points,
        "phi_c": [0.03]*points,
    }


class ConstituentMassGrowthTests(unittest.TestCase):
    def test_growth_path_updates_and_reapplies_fractions(self):
        simulation = (ROOT / "python_codes/LV_simulation/LV_simulation.py").read_text()
        growth = (ROOT / "python_codes/LV_simulation/growth/growth.py").read_text()
        self.assertIn("incremental_growth_volume_ratios()", simulation)
        self.assertIn("update_constituent_masses(", simulation)
        self.assertIn("apply_constituent_fractions(self.mesh)", simulation)
        self.assertIn("apply_constituent_fractions(self.gr.mechan)", simulation)
        self.assertIn("functions['theta_fiber']*functions['theta_sheet']", growth)

    def test_proportional_allocation_preserves_baseline_fractions(self):
        bookkeeper = MASS.ConstituentMassBookkeeper(
            initial_fractions(), 2,
            {"enabled": [True], "density": [1.0],
             "positive_growth_allocation": {"mode": ["proportional"]}})
        result = bookkeeper.update_positive_growth(
            [1.10, 1.25], [1.10, 1.25])
        for value in result["fractions"]["phi_m"]:
            self.assertAlmostEqual(value, 0.70)
        for value in result["fractions"]["phi_g"]:
            self.assertAlmostEqual(value, 0.27)
        for value in result["fractions"]["phi_c"]:
            self.assertAlmostEqual(value, 0.03)

    def test_collagen_enriched_allocation_increases_collagen_fraction(self):
        bookkeeper = MASS.ConstituentMassBookkeeper(
            initial_fractions(points=1), 1,
            {"enabled": [True], "density": [1.0],
             "positive_growth_allocation": {
                 "mode": ["fixed"], "myofiber": [0.20],
                 "ground_matrix": [0.10], "collagen": [0.70]}})
        result = bookkeeper.update_positive_growth([1.20], [1.20])
        self.assertGreater(result["fractions"]["phi_c"][0], 0.03)
        self.assertAlmostEqual(
            sum(result["fractions"][key][0]
                for key in ("phi_m", "phi_g", "phi_c")), 1.0)

    def test_total_mass_matches_positive_growth_volume(self):
        density = 1.05
        bookkeeper = MASS.ConstituentMassBookkeeper(
            initial_fractions(points=1), 1,
            {"enabled": [True], "density": [density],
             "positive_growth_allocation": {
                 "mode": ["fixed"], "myofiber": [0.20],
                 "ground_matrix": [0.10], "collagen": [0.70]}})
        old_volume = 2.0
        result = bookkeeper.update_positive_growth([1.25], [1.20])
        new_volume = old_volume*1.20
        total_mass = sum(result["referential_density"][name][0]
                         for name in MASS.CONSTITUENTS)*new_volume
        self.assertAlmostEqual(result["added_volume_ratio"][0], 0.25)
        self.assertAlmostEqual(total_mass, density*(old_volume + 0.5))

    def test_nonpositive_growth_does_not_remove_mass(self):
        bookkeeper = MASS.ConstituentMassBookkeeper(
            initial_fractions(points=1), 1,
            {"enabled": [True],
             "positive_growth_allocation": {"mode": ["proportional"]}})
        before = bookkeeper.fractions()
        result = bookkeeper.update_positive_growth([0.9], [1.0])
        self.assertEqual(result["added_volume_ratio"], [0.0])
        self.assertEqual(result["fractions"], before)

    def test_no_growth_with_identity_ale_is_invariant(self):
        bookkeeper = MASS.ConstituentMassBookkeeper(
            initial_fractions(points=1), 1,
            {"enabled": [True], "density": [1.0],
             "positive_growth_allocation": {"mode": ["proportional"]}})
        before = dict((name, list(values)) for name, values in
                      bookkeeper.referential_density.items())
        result = bookkeeper.update_positive_growth([1.0], [1.0])
        self.assertEqual(result["referential_density"], before)

    def test_repeated_growth_preserves_mass_balance(self):
        density = 1.05
        bookkeeper = MASS.ConstituentMassBookkeeper(
            initial_fractions(points=1), 1,
            {"enabled": [True], "density": [density],
             "positive_growth_allocation": {
                 "mode": ["fixed"], "myofiber": [0.20],
                 "ground_matrix": [0.10], "collagen": [0.70]}})
        volume = 3.0
        expected_mass = density*volume
        for growth_ratio, ale_ratio in ((1.10, 1.08), (1.05, 1.04)):
            expected_mass += density*volume*(growth_ratio - 1.0)
            volume *= ale_ratio
            result = bookkeeper.update_positive_growth(
                [growth_ratio], [ale_ratio])
            represented_mass = sum(
                result["referential_density"][name][0]
                for name in MASS.CONSTITUENTS)*volume
            self.assertAlmostEqual(represented_mass, expected_mass)

    def test_growth_uses_stable_material_point_keys(self):
        growth = (ROOT / "python_codes/LV_simulation/growth/growth.py").read_text()
        self.assertIn("global_indices(topological_dimension)", growth)
        self.assertIn("tabulate_local_to_global_dofs()", growth)
        self.assertIn("ownership_range()", growth)
        self.assertIn("set(target_keys) != set(self.material_point_keys)", growth)
        self.assertIn("order_spatial_mass_data_for_mesh", growth)

    def test_validation_inputs_preserve_baseline_material_parameters(self):
        baseline = json.loads((ROOT / "demos/base/sim_inputs/base_instruction.json").read_text())
        baseline_passive = baseline["mesh"]["forms_parameters"][
            "passive_law_parameters"]
        baseline_hs = baseline["model"]["half_sarcomere"]
        baseline_circulation = baseline["model"]["circulation"]
        for name in ("no_growth", "proportional", "collagen_enriched"):
            candidate = json.loads((ROOT / "validation/configs" /
                ("growth_mass_validation_%s.json" % name)).read_text())
            self.assertEqual(candidate["mesh"]["forms_parameters"][
                "passive_law_parameters"], baseline_passive)
            self.assertEqual(candidate["model"]["half_sarcomere"], baseline_hs)
            self.assertEqual(candidate["model"]["circulation"], baseline_circulation)

    def test_updated_fractions_feed_production_forms(self):
        forms = (ROOT / "python_codes/LV_simulation/dependencies/forms.py").read_text()
        self.assertIn("phi_g*W_ground + phi_m*W_myo + phi_c*W_collagen", forms)


if __name__ == "__main__":
    unittest.main()
