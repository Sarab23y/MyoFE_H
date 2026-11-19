"""Regression tests for the JSON Unicode -> legacy DOLFIN filename boundary."""
import ast
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
from contextlib import redirect_stdout

from validation import diagnostic_io
from validation.validate_myosim_mixed_quadrature_ordering import _production_mesh, resolve_mesh_path


class HDF5PathCompatibilityTests(unittest.TestCase):
    def test_python3_text_bytes_and_pathlike_become_native_text(self):
        for value in ('/tmp/mesh.h5', b'/tmp/mesh.h5', Path('/tmp/mesh.h5')):
            self.assertIs(type(diagnostic_io.native_filename(value)), str)
            self.assertEqual(diagnostic_io.native_filename(value),
                             str(value) if isinstance(value, Path) else '/tmp/mesh.h5')
        self.assertEqual(diagnostic_io.native_filename('/tmp/ventricle_\u03b1.h5'.encode('utf-8')),
                         '/tmp/ventricle_\u03b1.h5')

    def test_emulated_legacy_swig_rejects_unicode_then_accepts_normalized_path(self):
        comm = types.SimpleNamespace(Get_rank=lambda: 0)
        def strict_binding(actual_comm, filename, mode):
            self.assertIs(actual_comm, comm)
            self.assertEqual(mode, 'r')
            if type(filename) is not bytes:
                raise TypeError("in method 'new_HDF5File', argument 2 of type 'std::string const'")
            return filename
        value = json.loads('{"mesh_path":["/tmp/mesh.h5"]}')['mesh_path'][0]
        with self.assertRaises(TypeError):
            strict_binding(comm, value, 'r')
        with mock.patch.object(diagnostic_io, 'PYTHON2', True), redirect_stdout(io.StringIO()):
            self.assertEqual(diagnostic_io.open_hdf5_file(strict_binding, comm, value), b'/tmp/mesh.h5')

    def test_legacy_encoding_matches_working_production_recode(self):
        source = Path(__file__).resolve().parents[1]/'python_codes/LV_simulation/dependencies/recode_dictionary.py'
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_recode_value')
        namespace = {'unicode': str}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<production recode>', 'exec'), namespace)
        value = '/tmp/ventricle_\u03b1.h5'
        with mock.patch.object(diagnostic_io, 'PYTHON2', True):
            self.assertEqual(diagnostic_io.native_filename(value), namespace['_recode_value'](value))
            self.assertEqual(diagnostic_io.native_filename(b'/tmp/mesh.h5'), b'/tmp/mesh.h5')

    def test_relative_json_path_and_constructor_types_through_shared_loader(self):
        for legacy in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                (repo/'python_codes').mkdir()
                target = repo/'mesh.h5'
                target.write_bytes(b'test fixture')
                instruction = repo/'instruction.json'
                instruction.write_text(json.dumps({'mesh': {'mesh_path': ['../mesh.h5']}}))
                comm = types.SimpleNamespace(Get_rank=lambda: 0)
                events = []
                class Handle:
                    def read(self, mesh, dataset, distributed):
                        events.append(('read', dataset, distributed))
                    def close(self):
                        events.append(('close',))
                def constructor(actual_comm, filename, mode):
                    self.assertIs(actual_comm, comm)
                    self.assertIs(type(filename), bytes if legacy else str)
                    self.assertEqual(os.fsdecode(filename), str(target.resolve()))
                    return Handle()
                with mock.patch.object(diagnostic_io, 'PYTHON2', legacy), redirect_stdout(io.StringIO()):
                    unused_mesh, path = _production_mesh(str(instruction), str(repo), (constructor, object, lambda: comm))
                self.assertEqual(path, resolve_mesh_path(str(instruction), str(repo)))
                self.assertEqual(events, [('read', 'ellipsoidal', False), ('close',)])

    def test_invalid_filename_and_constructor_errors_are_explicit(self):
        for path in (None, 7, [], '', 'bad\x00name.h5'):
            with self.assertRaises((TypeError, ValueError)):
                diagnostic_io.native_filename(path)
        comm = types.SimpleNamespace(Get_rank=lambda: 0)
        def fail(*args):
            raise TypeError('argument 1 communicator mismatch')
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'argument 1 communicator mismatch.*filename_type'):
                diagnostic_io.open_hdf5_file(fail, comm, '/tmp/mesh.h5')

    def test_constructor_log_exposes_types_and_unchanged_signature(self):
        comm = types.SimpleNamespace(Get_rank=lambda: 0)
        capture = io.StringIO()
        with redirect_stdout(capture):
            diagnostic_io.open_hdf5_file(lambda *args: object(), comm, '/tmp/mesh.h5')
        payload = json.loads(capture.getvalue().split('] ', 1)[1])
        self.assertEqual(payload['filename_type'], 'str')
        self.assertEqual(payload['signature'], 'HDF5File(comm, filename, mode)')
        self.assertIn('communicator_type', payload)


if __name__ == '__main__':
    unittest.main()
