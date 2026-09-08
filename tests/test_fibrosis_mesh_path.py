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

    def test_mesh_diagnostic_uses_mpi4py_communicator_for_collectives(self):
        diagnostic = MESH_SOURCE.split(
            "    def report_mesh_diagnostics(self):", 1)[1]
        diagnostic = diagnostic.split("    def initialize_functions", 1)[0]
        self.assertIn("mpi_comm = self._get_mpi4py_comm()", diagnostic)
        for invalid_call in (
                "self.comm.allreduce", "self.comm.allgather",
                "self.comm.reduce", "self.comm.gather",
                "self.comm.Get_size()", "self.comm.Get_rank()"):
            self.assertNotIn(invalid_call, diagnostic)
        self.assertIn("op=MPI4PY.SUM", diagnostic)
        self.assertIn("op=MPI4PY.MIN", diagnostic)

    def test_no_heterogeneous_parameter_dictionary_debug_dump(self):
        source = (ROOT / (
            "python_codes/LV_simulation/dependencies/"
            "assign_heterogeneous_params.py")).read_text()
        self.assertNotIn("print('het_dolfin_dict')", source)

    def test_spatial_displacement_uses_vector_quadrature_space(self):
        simulation_source = (ROOT / (
            "python_codes/LV_simulation/LV_simulation.py")).read_text()
        writer = simulation_source.split(
            "    def write_complete_data_to_spatial_sim_data(self,rank):",
            1)[1]
        writer = writer.split("    def check_output_directory_folder", 1)[0]
        self.assertNotIn("['fiber_FS']", writer)
        self.assertIn(
            "'function_spaces']['material_coord_system_space']", writer)
        self.assertIn(
            "self.mesh.model['functions']['w'].sub(0),\n"
            "                displacement_output_space", writer)


if __name__ == "__main__":
    unittest.main()
