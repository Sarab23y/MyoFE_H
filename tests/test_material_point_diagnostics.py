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

    def test_displacement_script_is_python3_lcc_and_refuses_overwrite(self):
        source = read('validation/compare_cycle_displacements.py')
        self.assertTrue(source.startswith('#!/usr/bin/env python3'))
        self.assertIn("attr.get('Name','')", source)
        self.assertIn("if name not in found", source)
        self.assertIn("raise ValueError('conflicting %s records", source)
        self.assertIn('refusing to overwrite', source)
        self.assertIn('rigid(', source)
        self.assertIn('deformation_gradient', source)
        self.assertIn('quadratic_tetra10', source)
        self.assertIn("interpolate(displacement,target,cyc)", source)
        self.assertIn("interpolate(active,target,cyc)", source)
        self.assertIn('active_displacement_difference_correlation', source)
        self.assertIn('containing_tetra', source)
        self.assertIn('source_files_modified', source)

    def test_quadrature_ids_are_mapped_through_coordinates(self):
        source = read('validation/compare_cycle_displacements.py')
        self.assertIn("q=np.load(args.quadrature_coordinates)", source)
        compact = ''.join(source.split())
        self.assertIn('containing_tetra(coordinate,points,cells', compact)
        self.assertNotIn('points[global_id]', source)

    def test_displacement_launcher_is_read_only_and_job_specific(self):
        source = read('validation/run_cycle_displacement_analysis.slurm')
        self.assertIn('displacement_cycle_analysis_${SLURM_JOB_ID}', source)
        self.assertIn('ECM_40s_healthy_quad_validation_optimized', source)
        self.assertIn('mesh_output/solution.xdmf', source)
        self.assertIn('sim_output/quadrature_dof.npy', source)
        self.assertIn("-c 'import numpy, h5py'", source)
        self.assertIn('"$PYTHON_BIN" "$REPO/validation/compare_cycle_displacements.py"', source)
        self.assertNotIn('MyoFE.py', source)


if __name__ == '__main__':
    unittest.main()
