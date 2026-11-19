"""Compatibility contracts only; fixture tests do not certify LCC FEniCS."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_builder_2017',str(ROOT/'validation/build_native_newton_observer.py'))
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class Dolfin2017CompatibilityTests(unittest.TestCase):
    def test_2017_and_2019_have_explicit_exact_symbol_contracts(self):
        for version in ('2017.1.0','2019.1.0'):
            got=builder.select_symbols(version,['c','u'],[builder.EXPECTED['CONVERGED'],builder.EXPECTED['UPDATE']])
            self.assertEqual(got,{'CONVERGED':'c','UPDATE':'u'})
        for version in ('2017.2.0','2019.1.0.post0','0.9.0'):
            with self.assertRaises(RuntimeError):builder.select_symbols(version,[],[])

    def test_changed_missing_and_ambiguous_signatures_fail_closed(self):
        c,u=builder.EXPECTED['CONVERGED'],builder.EXPECTED['UPDATE']
        for symbols,names in [(['c'],[c]),(['c','u'],[c,u.replace('double, ','')]),
                              (['c','u','v'],[c,u,u]),(['c','u'],[c,u.replace('unsigned long','unsigned int')])]:
            with self.assertRaises(RuntimeError):builder.select_symbols('2017.1.0',symbols,names)

    def test_python2_demangling_uses_communicate_and_reports_errors(self):
        process=mock.Mock(returncode=0);process.communicate.return_value=(b'first\nsecond\n',b'')
        with mock.patch.object(builder.subprocess,'Popen',return_value=process) as popen:
            self.assertEqual(builder.demangle(['_c','_u']),['first','second'])
            process.communicate.assert_called_once_with(b'_c\n_u\n')
            self.assertEqual(popen.call_args[0][0],['c++filt'])
        process.returncode=1;process.communicate.return_value=(b'',b'failure')
        with mock.patch.object(builder.subprocess,'Popen',return_value=process):
            with self.assertRaises(RuntimeError):builder.demangle(['c'])

    def build_fixture(self, fail_contract=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);library=root/'libdolfin.so';library.write_bytes(b'SYNTHETIC library, not ELF')
            target=root/'new_build'
            def output(command):
                if command[:2]==['pkg-config','--variable=libdir']:return str(root).encode()
                if command[0]=='nm':return b'0 T c\n0 T u\n'
                if command[0]=='pkg-config':return b'-I/mock/include -L/mock/lib -ldolfin'
                return b'fixture compiler'
            calls=[]
            def compile_fixture(command):
                calls.append(command)
                if '-fsyntax-only' in command and fail_contract:raise subprocess.CalledProcessError(1,command)
                if '-shared' in command:Path(command[command.index('-o')+1]).write_bytes(b'SYNTHETIC hook')
            with mock.patch.object(builder.sys,'platform','linux'),mock.patch.object(builder.platform,'machine',return_value='x86_64'),mock.patch.dict(sys.modules,{'dolfin':types.SimpleNamespace(__version__='2017.1.0')}),mock.patch.object(builder.subprocess,'check_output',side_effect=output),mock.patch.object(builder,'demangle',return_value=[builder.EXPECTED['CONVERGED'],builder.EXPECTED['UPDATE']]),mock.patch.object(builder.subprocess,'check_call',side_effect=compile_fixture):
                if fail_contract:
                    with self.assertRaises(subprocess.CalledProcessError):builder.build(str(target))
                    self.assertFalse((target/'build.json').exists());self.assertEqual(len(calls),1)
                else:
                    builder.build(str(target));report=json.loads((target/'build.json').read_text())
                    self.assertEqual(report['adapter_profile'],'dolfin-2017.1-native-v1')
                    self.assertTrue(report['header_contract_compiled'])
                    self.assertFalse(report['callback_reachability_validated'])
                    self.assertFalse(report['physical_invariance_validated'])
                    self.assertIn('-fsyntax-only',calls[0]);self.assertIn('-Wl,-z,defs',calls[1])
                    with self.assertRaises(FileExistsError):builder.build(str(target))

    def test_mocked_2017_build_checks_headers_before_link_and_preserves_runtime_gates(self):
        self.build_fixture()

    def test_header_contract_failure_never_generates_success_report(self):
        self.build_fixture(True)

    def test_cpp_contract_requires_virtual_hooks_and_native_accessors(self):
        self.assertEqual(builder.CONTRACT.count(' override;'),2)
        for name in ('residual()','relative_residual()','krylov_iterations()'):
            self.assertIn(name,builder.CONTRACT)
        cpp=(ROOT/'validation/native_newton_observer.cpp').read_text()
        self.assertEqual(cpp.count('forward(this, r, problem, iteration)'),1)
        self.assertEqual(cpp.count('forward(this, x, dx, relaxation, problem, iteration)'),1)
        self.assertNotIn('x.axpy',cpp)

    def test_checkpoint_reader_uses_existing_legacy_communicator_helper(self):
        reader=(ROOT/'validation/validate_fe_cycle_checkpoint.py').read_text()
        self.assertIn('mesh=Mesh(world_comm(df))',reader)
        self.assertIn('comm=as_mpi4py(world_comm(df))',reader)
        self.assertNotIn('Mesh(MPI.comm_world)',reader)
        smoke=(ROOT/'validation/smoke_cg2_checkpoint.py').read_text()
        self.assertIn('exact_mixed_coefficient_roundtrip=exact',smoke)
        self.assertIn("'CG',cell,2",smoke)

    def test_probe_failure_is_coordinated_before_collectives(self):
        source=(ROOT/'validation/smoke_native_newton.py').read_text()
        self.assertLess(source.index('if any(comm.allgather(injected))'),source.index('parts=comm.allgather'))
        self.assertIn('one_rank_observer_failure_deferred_and_solution_unchanged=True',source)

    def test_preflight_uses_authoritative_runtime_and_all_rank_counts(self):
        source=(ROOT/'validation/run_fe_cycle_preflight.slurm').read_text()
        self.assertIn('dolfin.__version__ == "2017.1.0"',source)
        self.assertIn('for N in 1 32;',source)
        self.assertIn('--partition=CAC48M192_L',source)
        self.assertIn('test ! -e "$OUT"',source)
        complete=(ROOT/'validation/validate_lcc.slurm').read_text()
        self.assertIn('--partition=CAC48M192_L',complete)
        self.assertNotIn('--mode full',complete)

if __name__=='__main__':unittest.main()
