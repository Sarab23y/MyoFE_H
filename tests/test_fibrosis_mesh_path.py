import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MESH_SOURCE = (
    ROOT / "python_codes/LV_simulation/mesh/mesh.py").read_text()


class FibrosisMeshExecutionPathTests(unittest.TestCase):
    def test_weak_form_has_no_obsolete_tensor_space_projection(self):
        weak_form = MESH_SOURCE.split("    def create_weak_form(self):", 1)[1]
        weak_form = weak_form.split("    def initialize_dolfin_functions", 1)[0]
        self.assertNotIn("['tensor_space']", weak_form)

    def test_static_disarray_is_not_called_during_initialization(self):
        initialization = MESH_SOURCE.split(
            "    def apply_static_fiber_architecture", 1)[0]
        self.assertNotIn("self.apply_static_fiber_architecture(", initialization)

    def test_current_scaler_spelling_maps_to_scalar_space(self):
        self.assertIn(
            "fcn_spaces['scalar'] = fcn_spaces['scaler']", MESH_SOURCE)

    def test_mesh_diagnostic_is_read_only(self):
        diagnostic = MESH_SOURCE.split(
            "    def report_mesh_diagnostics(self):", 1)[1]
        diagnostic = diagnostic.split("    def initialize_functions", 1)[0]
        for mutation in ("set_local", "vector()[:]", "ALE.move", "np.random"):
            self.assertNotIn(mutation, diagnostic)


if __name__ == "__main__":
    unittest.main()
