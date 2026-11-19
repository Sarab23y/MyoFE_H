"""Actual CLI/import checks plus isolated workflow-failure regression tests."""
import ast
import builtins
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from validation import preflight_diagnostics as preflight
from validation.diagnostic_io import root_call, write_report
from validation.validate_myosim_mixed_quadrature_ordering import _state_component_count
from validation.validate_ecm_quadrature_mpi import compare_reports
from validation.compare_cycle_displacements import coordinate_lookup, read_xdmf

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticEntrypointTests(unittest.TestCase):
    def test_xdmf_reader_loads_only_requested_cycle_window(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with h5py.File(str(directory/'frames.h5'), 'w') as handle:
                handle['mesh'] = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
                handle['cells'] = np.array([[0,1,2,3]])
                for time in (1,2,3):
                    handle['u%d' % time] = np.ones((4,3))*time
            grids = []
            for time in (1,2,3):
                grids.append('<Grid><Topology TopologyType="Tetrahedron"><DataItem Format="HDF">frames.h5:/cells</DataItem></Topology>'
                             '<Geometry><DataItem Format="HDF">frames.h5:/mesh</DataItem></Geometry>'
                             '<Time Value="%d"/><Attribute Name="displacement" Center="Node">'
                             '<DataItem Format="HDF">frames.h5:/u%d</DataItem></Attribute></Grid>' % (time,time))
            path = directory/'frames.xdmf'
            path.write_text('<Xdmf><Domain>'+''.join(grids)+'</Domain></Xdmf>')
            records = read_xdmf(str(path), ('displacement',), (2.,2.))[3]['displacement']
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0][0], 2.)
            np.testing.assert_equal(records[0][1], np.ones((4,3))*2.)

    def test_visualization_mapping_uses_saved_ids_not_array_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinates = Path(directory)/'quadrature_dof.npy'
            ids = Path(directory)/'quadrature_global_dof.npy'
            np.save(coordinates, [[1., 2., 3.], [4., 5., 6.]])
            np.save(ids, [4844, 1580])
            result = coordinate_lookup(str(coordinates))
            np.testing.assert_equal(result[1580], [4., 5., 6.])
            np.save(ids, [4844, 4844])
            with self.assertRaises(ValueError):
                coordinate_lookup(str(coordinates))

    def test_real_direct_and_package_cli_from_foreign_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in preflight.MAPPING + ('preflight_diagnostics',):
                direct = subprocess.run([sys.executable, str(ROOT/'validation'/(name+'.py')), '--help'],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(direct.returncode, 0, direct.stderr)
                package = subprocess.run([sys.executable, '-m', 'validation.'+name, '--help'],
                                         cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(package.returncode, 0, package.stderr)

    def test_python2_style_relative_import_valueerror_is_avoided(self):
        original = builtins.__import__

        def old_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level and globals and not globals.get('__package__'):
                raise ValueError('Attempted relative import in non-package')
            return original(name, globals, locals, fromlist, level)

        with mock.patch.object(sys, 'path', [str(ROOT/'validation')]+sys.path):
            with mock.patch.object(builtins, '__import__', side_effect=old_import):
                for name in preflight.MAPPING + preflight.SHARED + preflight.ANALYSIS:
                    runpy.run_path(str(ROOT/'validation'/(name+'.py')), run_name='standalone_import_test')

    def test_state_count_matches_production_setup_methods(self):
        # Execute only the actual production setup method bodies. Importing the
        # complete cellular package would hide unnecessary dependency coupling.
        source = ast.parse((ROOT/'python_codes/LV_simulation/half_sarcomere/myofilaments/myofilaments.py').read_text())
        klass = next(node for node in source.body if isinstance(node, ast.ClassDef))
        functions = [node for node in klass.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('set_up_3_state_with_SRX', 'set_up_4_state_with_SRX')]
        namespace = {'np': np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), '<production setup>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.json'
            for states in (3, 4):
                for suffix in ('', '_and_exp_detach'):
                    scheme = '%d_state_with_SRX%s' % (states, suffix)
                    impl = dict(bin_min=[-10], bin_max=[10], bin_width=[.5], kinetic_scheme=[scheme])
                    path.write_text(json.dumps({'model': {'half_sarcomere': {'myofilaments': {'implementation': impl}}}}))
                    actual = types.SimpleNamespace(no_of_x_bins=len(np.arange(-10, 10.5, .5)),
                                                   data={}, return_n_overlap=lambda: 1., update_data=lambda: None)
                    namespace['set_up_%d_state_with_SRX' % states](actual)
                    self.assertEqual(_state_component_count(path, None), len(actual.y))

    def test_reports_refuse_overwrite_and_remove_invalid_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            write_report(path, {'passed': True})
            with self.assertRaises(OSError):
                write_report(path, {'passed': False})
            self.assertTrue(json.loads(path.read_text())['passed'])
            path.unlink()
            with self.assertRaises(ValueError):
                write_report(path, {'value': float('nan')})
            self.assertFalse(path.exists())

    def test_root_io_failure_is_broadcast_to_other_ranks(self):
        delivered = []
        root = types.SimpleNamespace(Get_rank=lambda: 0, bcast=lambda value, root=0: delivered.append(value) or value)
        with self.assertRaisesRegex(RuntimeError, 'disk failure'):
            root_call(root, lambda: (_ for _ in ()).throw(IOError('disk failure')))
        other = types.SimpleNamespace(Get_rank=lambda: 7, bcast=lambda value, root=0: delivered[0])
        with self.assertRaisesRegex(RuntimeError, 'disk failure'):
            root_call(other, lambda: self.fail('non-root callback executed'))

    def test_scalar_comparison_rejects_failed_or_empty_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b, out = [Path(directory)/name for name in ('a.json', 'b.json', 'out.json')]
            for path in (a, b):
                path.write_text(json.dumps({'passed': True, 'points': []}))
            self.assertEqual(compare_reports(a, b, out), 1)
            self.assertFalse(json.loads(out.read_text())['passed'])

    def test_orchestrator_runs_all_exact_entrypoints_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            seen, checks = [], []

            def fake_command(stage, argv, cwd, output, checks):
                seen.append((stage, argv))
                checks.append({'stage': stage, 'returncode': 0})
                path = argv[argv.index('--output')+1]
                write_report(path, {'passed': True})

            with mock.patch.object(preflight, 'command', side_effect=fake_command):
                preflight.workflow(str(ROOT), 'input.json', 'saved', directory, [1,32], checks)
            self.assertEqual(len(seen), 10)
            for stage, argv in seen[:8]:
                self.assertEqual(argv[:2], ['mpiexec', '-np'])
                self.assertTrue(argv[4].endswith('.py'))
            self.assertNotIn('--saved-output', seen[3][1])
            self.assertIn('--saved-output', seen[7][1])
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(preflight, 'command', side_effect=RuntimeError('child failed')) as called:
                with self.assertRaisesRegex(RuntimeError, 'child failed'):
                    preflight.workflow(str(ROOT), 'input.json', 'saved', directory, [1,32], [])
                self.assertEqual(called.call_count, 1)

    def test_subprocess_failure_is_recorded_with_log(self):
        with tempfile.TemporaryDirectory() as directory:
            checks = []
            with self.assertRaisesRegex(RuntimeError, 'exit 4'):
                preflight.command('failure', [sys.executable, '-c', 'raise SystemExit(4)'],
                                  directory, directory, checks)
            self.assertEqual(checks[0]['returncode'], 4)
            self.assertTrue(Path(checks[0]['log']).exists())

    def test_preflight_failure_report_and_retry_protection(self):
        instruction = ROOT/'demos/ECM/sim_input/base_instruction_Br_ECM_40s_healthy_quad_validation_optimized.json'
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory)/'simulation/sim_output'
            saved.mkdir(parents=True)
            output = Path(directory)/'preflight'
            argv = ['--repo', str(ROOT), '--input-json', str(instruction), '--saved-output', str(saved),
                    '--output', str(output), '--static-only']
            with mock.patch.object(preflight, 'resolve_mesh_path', return_value='test-mesh'):
                with mock.patch.object(preflight, 'command', side_effect=RuntimeError('missing dependency')):
                    self.assertEqual(preflight.main(argv), 1)
            report = json.loads((output/'preflight.json').read_text())
            self.assertFalse(report['passed'])
            self.assertFalse(report['mapping_workflow_validated'])
            self.assertEqual(report['validated_ranks'], [])
            original = (output/'preflight.json').read_bytes()
            with self.assertRaises(SystemExit):
                preflight.main(argv)
            self.assertEqual((output/'preflight.json').read_bytes(), original)

    def test_invalid_cli_arguments_fail_before_runtime(self):
        script = ROOT/'validation/preflight_diagnostics.py'
        result = subprocess.run([sys.executable, str(script), '--ranks', '2'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)


if __name__ == '__main__':
    unittest.main()
