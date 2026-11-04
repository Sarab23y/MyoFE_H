import ast
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(relative):
    with open(os.path.join(ROOT, relative)) as handle:
        return handle.read()


class MaterialPointDiagnosticTests(unittest.TestCase):
    def test_mixed_ordering_test_is_python2_syntax_compatible(self):
        source = read(
            'validation/validate_myosim_mixed_quadrature_ordering.py')
        self.assertIn('from __future__ import division, print_function', source)
        self.assertIn('mixed_space.sub(component).dofmap().cell_dofs', source)
        self.assertIn('scalar_local * components + component', source)
        self.assertIn('forward_mismatch_count', source)
        self.assertIn('inverse_mismatch_count', source)
        self.assertIn('_production_mesh', source)
        self.assertIn('production_local_points == scalar_owned', source)

    def test_ordering_launcher_runs_one_and_thirty_two_ranks(self):
        source = read('validation/run_myosim_quadrature_ordering.slurm')
        self.assertIn('mpiexec -np 1 python', source)
        self.assertIn('mpiexec -np 32 python', source)
        self.assertIn('CAC48M192_L', source)
        self.assertIn('cascadeb022', source)
        self.assertLess(source.index('singularity exec'),
                        source.index('mpiexec -np 1'))

    def test_displacement_script_is_python3_and_refuses_overwrite(self):
        source = read('validation/compare_cycle_displacements.py')
        ast.parse(source)
        self.assertIn('output.mkdir(parents=True, exist_ok=False)', source)
        self.assertIn('best_fit_rigid', source)
        self.assertIn('affine_tetra_gradient', source)
        self.assertIn('containing_tetra', source)
        self.assertIn('phase_selection_passed', source)
        self.assertIn('source_files_modified', source)

    def test_quadrature_ids_are_mapped_through_coordinates(self):
        source = read('validation/compare_cycle_displacements.py')
        self.assertIn('quadrature_dof.npy ordered by global q-DOF', source)
        self.assertIn('containing_tetra(coordinate, points, tetrahedra)', source)
        self.assertNotIn('points[global_id]', source)


if __name__ == '__main__':
    unittest.main()
