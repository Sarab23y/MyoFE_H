#!/usr/bin/env python
"""Run diagnostic entry points in isolated directories before a mapping job.

Python 2.7/3 compatible. The orchestration process never initializes MPI.
DOLFIN imports, compilation and MPI tests execute in separate child processes.
"""
from __future__ import print_function
import argparse
import json
import os
import subprocess
import sys

if __package__:
    from .diagnostic_io import write_report
    from .validate_myosim_mixed_quadrature_ordering import resolve_mesh_path
else:
    from diagnostic_io import write_report
    from validate_myosim_mixed_quadrature_ordering import resolve_mesh_path

MAPPING = ('mpi_compat', 'validate_ecm_quadrature_mpi',
           'validate_myosim_mixed_quadrature_ordering', 'export_verified_quadrature_geometry')
SHARED = ('diagnostic_io', 'preflight_diagnostics')
ANALYSIS = ('analyze_phase_region', 'analyze_ecm_spatial_anomalies',
            'compare_cycle_displacements')

def command(stage, argv, cwd, output, checks):
    log = os.path.join(output, stage + '.log')
    with open(log, 'w') as handle:
        status = subprocess.call(argv, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    checks.append({'stage': stage, 'command': argv, 'returncode': status, 'log': log})
    if status:
        with open(log, 'rb') as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell()-12000))
            detail = handle.read().decode('utf-8', 'replace')
        raise RuntimeError('%s failed (exit %d). See %s\n%s' % (stage, status, log, detail))

def require_passed(path):
    with open(path) as handle:
        if json.load(handle).get('passed') is not True:
            raise ValueError('Diagnostic reported failure: ' + path)

def workflow(repo, input_json, saved_output, output, ranks, checks):
    """Exact standalone entry points used by the production-rank diagnostic job."""
    for n in ranks:
        prefix = os.path.join(output, 'rank_%d' % n)
        os.mkdir(prefix)
        cases = [
            ('comm', 'mpi_compat', ['--expected-size', str(n)]),
            ('scalar', 'validate_ecm_quadrature_mpi', []),
            ('mixed', 'validate_myosim_mixed_quadrature_ordering', ['--input-json', input_json]),
            ('geometry', 'export_verified_quadrature_geometry', ['--input-json', input_json])]
        if n == 32:
            cases[-1][2].extend(['--saved-output', saved_output])
        for label, name, extra in cases:
            result = os.path.join(prefix, label + '.json')
            command('%s_%d' % (label, n),
                    ['mpiexec', '-np', str(n), sys.executable,
                     os.path.join(repo, 'validation', name+'.py')] + extra + ['--output', result],
                    os.path.join(repo, 'python_codes'), output, checks)
            require_passed(result)
    if 1 in ranks and 32 in ranks:
        for label, name in (('scalar', 'validate_ecm_quadrature_mpi'),
                            ('geometry', 'export_verified_quadrature_geometry')):
            result = os.path.join(output, label+'_comparison.json')
            command(label+'_comparison',
                    [sys.executable, os.path.join(repo, 'validation', name+'.py'), '--compare',
                     os.path.join(output, 'rank_1', label+'.json'),
                     os.path.join(output, 'rank_32', label+'.json'), '--output', result],
                    repo, output, checks)
            require_passed(result)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--saved-output', required=True)
    parser.add_argument('--output', required=True, help='new isolated directory outside saved results')
    parser.add_argument('--static-only', action='store_true', help='imports/dependencies/inputs/CLI, no FE solves')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1], choices=[1, 32])
    parser.add_argument('--analysis-python', help='optional separate Python >=3.6 with NumPy and h5py')
    args = parser.parse_args(argv)
    repo, output, saved = [os.path.realpath(p) for p in (args.repo, args.output, args.saved_output)]
    input_json = os.path.realpath(args.input_json)
    # A report path below simulation results would violate read-only analysis.
    run_root = os.path.dirname(saved)
    if output == run_root or output.startswith(run_root + os.sep):
        parser.error('--output must be outside the simulation directory')
    try:
        os.makedirs(output)
    except OSError as error:
        parser.error('Cannot create a new preflight directory: %s' % error)
    checks = []
    report = {'passed': False, 'validated_ranks': [], 'mapping_workflow_validated': False,
              'checks': checks, 'analysis_environment_checked': bool(args.analysis_python)}
    try:
        if not os.path.isfile(input_json) or not os.path.isdir(saved):
            raise ValueError('Submitted input or saved-output directory is missing')
        with open(os.path.join(repo, 'validation', 'run_phase_mapping_validation.slurm')) as handle:
            directives = [line.strip() for line in handle if line.startswith('#SBATCH')]
        required = ['#SBATCH --ntasks=32', '#SBATCH --nodes=1', '#SBATCH --mem=32G',
                    '#SBATCH --partition=SKY32M192_D', '#SBATCH --nodelist=skylake047']
        if any(item not in directives for item in required):
            raise ValueError('Mapping launcher configuration differs from reviewed resources')
        report['launcher_directives'] = directives
        mesh = resolve_mesh_path(input_json, repo)
        report['mesh'] = mesh
        for name in MAPPING + SHARED:
            path = os.path.join(repo, 'validation', name+'.py')
            if not os.path.isfile(path):
                raise ValueError('Missing diagnostic dependency: '+path)
            command('compile_'+name, [sys.executable, '-c',
                'import py_compile,sys; py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)',
                path, os.path.join(output, name+'.pyc')], repo, output, checks)
        # Explicitly exercise both import modes and direct CLI parsing. Child
        # processes are important: MPI must not be initialized in this parent.
        for name in MAPPING + SHARED:
            for package in (False, True):
                module = 'validation.'+name if package else name
                directory = repo if package else os.path.join(repo, 'validation')
                command('import_%s_%s' % (name, 'package' if package else 'direct'),
                        [sys.executable, '-c', 'import importlib; importlib.import_module(%r)' % module],
                        directory, output, checks)
            if name != 'diagnostic_io':
                command('help_'+name, [sys.executable, os.path.join(repo, 'validation', name+'.py'), '--help'],
                        repo, output, checks)
        dependency_code = """
import sys, json, importlib
names = ['numpy', 'dolfin', 'ufl', 'ffc', 'petsc4py', 'mpi4py']
modules = dict((name, importlib.import_module(name)) for name in names)
from petsc4py import PETSc
from mpi4py import MPI
print(json.dumps(dict(
    python=sys.version, executable=sys.executable,
    versions=dict((name, str(getattr(module, '__version__', 'unknown')))
                  for name, module in modules.items()),
    petsc=PETSc.Sys.getVersion(), mpi=MPI.Get_library_version()), indent=2))
"""
        command('runtime_dependencies', [sys.executable, '-c', dependency_code], repo, output, checks)
        metadata_code = """
import sys, os, json, numpy as np
from validation.validate_myosim_mixed_quadrature_ordering import _state_component_count
directory = sys.argv[1]
names = ['quadrature_global_dof', 'quadrature_dof', 'norm_dist_endo']
arrays = dict((name, np.load(os.path.join(directory, name+'.npy'))) for name in names)
ids = arrays[names[0]].reshape(-1)
coordinates = arrays[names[1]]
wall = arrays[names[2]].reshape(-1)
assert len(ids) == len(set(ids)) and coordinates.shape == (len(ids), 3), 'Geometry IDs/coordinates'
assert np.isfinite(ids).all() and (ids == ids.astype(np.int64)).all(), 'Noninteger/nonfinite IDs'
assert len(wall) == len(ids) and np.isfinite(coordinates).all(), 'Geometry shapes/values'
assert np.isfinite(wall).all() and ((wall >= 0) & (wall <= 1)).all(), 'Wall-depth values'
print('points', len(ids), 'MyoSim components', _state_component_count(sys.argv[2], None))
with open(sys.argv[2]) as handle:
    instruction = json.load(handle)
for space in instruction['mesh'].get('function_spaces', []):
    if space['name'][0] in ('quadrature_space', 'material_coord_system_space'):
        assert space['element_type'][0] == 'Quadrature' and space['degree'][0] == 2, 'Unsupported quadrature definition'
"""
        command('saved_metadata', [sys.executable, '-c', metadata_code, saved, input_json], repo, output, checks)
        if args.analysis_python:
            command('analysis_dependencies', [args.analysis_python, '-c',
                'import sys,numpy,h5py; assert sys.version_info >= (3,6); print(sys.version,numpy.__version__,h5py.__version__)'],
                repo, output, checks)
            for name in ANALYSIS:
                command('analysis_help_'+name, [args.analysis_python, os.path.join(repo,'validation',name+'.py'),'--help'],
                        repo, output, checks)
                command('analysis_import_'+name, [args.analysis_python, '-c',
                    'import importlib; importlib.import_module(%r)' % ('validation.'+name)],
                    repo, output, checks)
        if not args.static_only:
            workflow(repo, input_json, saved, output, args.ranks, checks)
            report['validated_ranks'] = args.ranks
        report['passed'] = True
        report['mapping_workflow_validated'] = 1 in report['validated_ranks'] and 32 in report['validated_ranks']
    except Exception as error:
        report['error'] = '%s: %s' % (type(error).__name__, error)
        print(report['error'], file=sys.stderr)
    report['unverified'] = (['MPI/FE ranks %d' % n for n in (1,32) if n not in report['validated_ranks']] +
                            ([] if args.analysis_python else ['separate Python 3 post-processing environment']))
    write_report(os.path.join(output, 'preflight.json'), report)
    print('Preflight report: '+os.path.join(output, 'preflight.json'))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
