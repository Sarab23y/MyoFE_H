#!/usr/bin/env python
"""Diagnostic-only production-mesh geometry/weight audit, Python 2.7 compatible.

Run with the production rank count to bind output IDs to their saved reference
coordinates. Serial/MPI comparison uses coordinates, never rank-specific IDs.
"""
from __future__ import division, print_function
import argparse
import hashlib
import json
import os
import subprocess

REPORTS = ('mpi_comm_1.json', 'mpi_comm_32.json', 'ecm_1.json', 'ecm_32.json', 'mixed_1.json', 'mixed_32.json',
           'geometry_1.json', 'geometry_32.json', 'ecm_comparison.json', 'geometry_comparison.json')
SOURCES = ('validation/validate_ecm_quadrature_mpi.py',
           'validation/mpi_compat.py',
           'validation/diagnostic_io.py',
           'validation/preflight_diagnostics.py',
           'validation/validate_myosim_mixed_quadrature_ordering.py',
           'validation/export_verified_quadrature_geometry.py',
           'validation/analyze_phase_region.py',
           'validation/analyze_ecm_spatial_anomalies.py',
           'diagnose_midwall_sarcomere_stdlib.py',
           'validation/run_phase_mapping_validation.slurm',
           'validation/run_phase_region_analysis.slurm')

if __package__:
    from .diagnostic_io import write_report, root_call
    from .mpi_compat import as_mpi4py, world_comm
    from .validate_myosim_mixed_quadrature_ordering import _production_mesh, _local_parent_map
else:
    from diagnostic_io import write_report, root_call
    from mpi_compat import as_mpi4py, world_comm
    from validate_myosim_mixed_quadrature_ordering import _production_mesh, _local_parent_map


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def check_reports(directory, input_json):
    """Reject mixed, stale or incomplete report suites before analysis."""
    reports = {}
    for name in REPORTS:
        with open(os.path.join(directory, name)) as handle:
            reports[name] = json.load(handle)
        if reports[name].get('passed') is not True:
            raise ValueError('mapping report did not pass: ' + name)
    expected_input = digest(input_json)
    meshes = set()
    for ranks in (1, 32):
        for kind in ('mpi_comm', 'ecm', 'mixed', 'geometry'):
            name = '%s_%d.json' % (kind, ranks)
            if reports[name].get('mpi_size') != ranks:
                raise ValueError('wrong MPI size: ' + name)
        for kind in ('mixed', 'geometry'):
            name = '%s_%d.json' % (kind, ranks)
            if reports[name].get('input_sha256') != expected_input:
                raise ValueError('submitted input mismatch: ' + name)
            mesh_hash = reports[name].get('mesh_sha256')
            if not mesh_hash:
                raise ValueError('missing mesh checksum: ' + name)
            meshes.add(mesh_hash)
        geom = reports['geometry_%d.json' % ranks]
        if geom.get('weights_verified') is not True or geom.get('vector_point_major_ordering_verified_all_ranks') is not True:
            raise ValueError('weights/vector mapping not verified')
        mixed_ranks = reports['mixed_%d.json' % ranks].get('ranks', [])
        if (len(mixed_ranks) != ranks or set(r.get('rank') for r in mixed_ranks) != set(range(ranks)) or
                not all(r.get('passed') is True for r in mixed_ranks)):
            raise ValueError('incomplete/failed mixed mapping rank reports')
    if any(reports[name].get('mesh_partitioner') != 'SCOTCH' for name in ('mixed_32.json', 'geometry_32.json')):
        raise ValueError('production MPI partitioner does not match SCOTCH')
    if len(meshes) != 1:
        raise ValueError('mapping reports used different meshes')
    if reports['geometry_32.json'].get('saved_mapping_verified') is not True:
        raise ValueError('production saved mapping not verified')
    return reports


def seal_suite(directory, input_json, image, repo):
    check_reports(directory, input_json)
    manifest = {'complete': True, 'schema_version': 1, 'input_sha256': digest(input_json),
                'image_sha256': digest(image),
                'repo_commit': subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD']).decode('ascii').strip(),
                'report_sha256': dict((name, digest(os.path.join(directory, name))) for name in REPORTS),
                'diagnostic_source_sha256': dict((name, digest(os.path.join(repo, name))) for name in SOURCES)}
    write_report(os.path.join(directory, 'mapping_manifest.json'), manifest)
    return manifest


def require_suite(directory, input_json, repo):
    with open(os.path.join(directory, 'mapping_manifest.json')) as handle:
        manifest = json.load(handle)
    if manifest.get('complete') is not True or manifest.get('schema_version') != 1:
        raise ValueError('mapping suite is not complete')
    if manifest.get('input_sha256') != digest(input_json):
        raise ValueError('mapping manifest submitted input mismatch')
    for key, names, root in (('report_sha256', REPORTS, directory),
                             ('diagnostic_source_sha256', SOURCES, repo)):
        hashes = manifest.get(key, {})
        if set(hashes) != set(names):
            raise ValueError('incomplete manifest: ' + key)
        for name in names:
            if digest(os.path.join(root, name)) != hashes[name]:
                raise ValueError('stale/modified mapping suite or diagnostic source: ' + name)
    check_reports(directory, input_json)
    return manifest


def compare(first, second, output):
    import numpy as np
    with open(first) as handle:
        a = json.load(handle)
    with open(second) as handle:
        b = json.load(handle)
    left, right = [sorted(r['points'], key=lambda p: p['coordinate']) for r in (a, b)]
    passed = (a['passed'] and b['passed'] and len(left) == len(right) and
              a['mesh_sha256'] == b['mesh_sha256'] and a['input_sha256'] == b['input_sha256'])
    error = 0.
    if passed:
        for x, y in zip(left, right):
            if not np.allclose(x['coordinate'], y['coordinate'], atol=1e-10, rtol=0):
                passed = False
            if not all(np.isfinite(p['reference_weight']) and p['reference_weight'] > 0 and
                       np.isfinite(p['wall_depth']) for p in (x, y)):
                passed = False
            error = max(error, abs(x['reference_weight']-y['reference_weight']),
                        abs(x['wall_depth']-y['wall_depth']))
        passed = passed and error <= 1e-10
    write_report(output, {'passed': bool(passed), 'maximum_weight_or_wall_difference': error,
                   'comparison': 'physical coordinates, reference weights and wall depth; IDs may differ'})
    return 0 if passed else 1


def run(args):
    import numpy as np
    import dolfin as df
    if as_mpi4py(world_comm(df)).Get_size() > 1:
        df.parameters['mesh_partitioner'] = 'SCOTCH'
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesh, mesh_path = _production_mesh(args.input_json, repo, (df.HDF5File, df.Mesh, lambda: world_comm(df)))
    raw_comm = mesh.mpi_comm()
    comm = as_mpi4py(raw_comm)
    element = df.FiniteElement('Quadrature', mesh.ufl_cell(), degree=2, quad_scheme='default')
    element._quad_scheme = 'default'
    space = df.FunctionSpace(mesh, element)
    wall = df.Function(space)
    h5 = df.HDF5File(raw_comm, mesh_path, 'r')
    # Fail if absent, unlike the production optional-read fallback.
    try:
        h5.read(wall, 'ellipsoidal/endo_dist')
    finally:
        h5.close()
    form = df.TestFunction(space)*df.dx(domain=mesh, metadata={'quadrature_degree': 2, 'quadrature_scheme': 'default'})
    weights = df.assemble(form, form_compiler_parameters={'representation': 'quadrature'}).get_local()
    n = len(wall.vector().get_local())
    coords = space.tabulate_dof_coordinates().reshape((-1, 3))[:n]
    wall_values = wall.vector().get_local()
    # Audit the separate VectorElement packing assumed by reshape((-1,3))
    # in immutable fiber-coordinate metadata, using authoritative cell maps.
    vector_element = df.VectorElement('Quadrature', mesh.ufl_cell(), degree=2, quad_scheme='default')
    vector_element._quad_scheme = 'default'
    vector_space = df.FunctionSpace(mesh, vector_element)
    vector_maps, vector_conflicts = _local_parent_map(mesh, space, vector_space, 3)
    vector_owned = len(df.Function(vector_space).vector().get_local())
    vector_coordinates = vector_space.tabulate_dof_coordinates().reshape((-1, 3))
    vector_mismatch = 0
    vector_seen = set()
    for component, component_map in enumerate(vector_maps):
        for scalar_dof, parent_dof in component_map.items():
            if scalar_dof < n:
                if parent_dof != scalar_dof*3 + component or parent_dof >= vector_owned:
                    vector_mismatch += 1
                elif not np.allclose(coords[scalar_dof], vector_coordinates[parent_dof], atol=1e-12, rtol=0):
                    vector_mismatch += 1
                vector_seen.add(parent_dof)
    vector_ok = not vector_conflicts and not vector_mismatch and len(vector_seen) == 3*n == vector_owned
    vector_ok_global = all(comm.allgather(bool(vector_ok)))
    metadata = {}
    for cell in df.cells(mesh):
        for q, dof in enumerate(space.dofmap().cell_dofs(cell.index())):
            if int(dof) < n:
                metadata[int(dof)] = {'cell_reference_vertices': mesh.coordinates()[cell.entities(0)].tolist(),
                                      'quadrature_point_in_cell': q}
    local = []
    for i in range(n):
        item = {'global_dof': int(space.dofmap().local_to_global_index(i)),
                'coordinate': coords[i].tolist(), 'wall_depth': float(wall_values[i]),
                'reference_weight': float(weights[i])}
        item.update(metadata[i]); local.append(item)
    gathered = comm.gather(local, root=0)
    reference_volume = float(df.assemble(df.Constant(1.)*df.dx(domain=mesh)))
    def finish():
        points = sorted([p for rank in gathered for p in rank], key=lambda p: p['global_dof'])
        xyz = np.asarray([p['coordinate'] for p in points])
        ids = [p['global_dof'] for p in points]
        w = np.asarray([p['reference_weight'] for p in points])
        wall = np.asarray([p['wall_depth'] for p in points])
        saved_verified = False
        hashes = {}
        mapping_error = None
        if args.saved_output:
            names = ('quadrature_dof.npy', 'quadrature_global_dof.npy', 'norm_dist_endo.npy')
            saved_xyz, saved_ids, saved_wall = [np.load(os.path.join(args.saved_output, name)) for name in names]
            saved_ids = np.asarray(saved_ids).reshape(-1)
            saved_wall = np.asarray(saved_wall).reshape(-1)
            index = dict((int(gid), i) for i, gid in enumerate(saved_ids))
            if len(index) == len(saved_ids) and set(ids) == set(index) and saved_xyz.shape == xyz.shape:
                order = [index[gid] for gid in ids]
                saved_verified = bool(np.allclose(xyz, saved_xyz[order], rtol=0, atol=1e-10) and
                                      np.allclose(wall, saved_wall[order], rtol=0, atol=1e-10))
            if not saved_verified:
                mapping_error = 'saved ID-coordinate/wall tuples differ from reconstructed production-rank space'
            hashes = dict((name, digest(os.path.join(args.saved_output, name))) for name in names)
        valid = bool(vector_ok_global and len(ids) == len(set(ids)) and len(set(tuple(c) for c in xyz)) == len(ids) and
                     np.isfinite(xyz).all() and np.isfinite(w).all() and np.all(w > 0) and
                     np.isfinite(wall).all() and np.all((wall >= 0) & (wall <= 1)) and
                     np.isclose(w.sum(), reference_volume, rtol=1e-10, atol=1e-12))
        report = {'passed': valid and (saved_verified if args.saved_output else True),
                  'saved_mapping_verified': saved_verified, 'mapping_error': mapping_error,
                  'weights_verified': valid, 'source_sha256': hashes,
                  'vector_point_major_ordering_verified_all_ranks': bool(vector_ok_global),
                  'mesh_sha256': digest(mesh_path), 'input_sha256': digest(args.input_json),
                  'mesh_path': mesh_path, 'mpi_size': int(comm.Get_size()),
                  'mesh_partitioner': str(df.parameters['mesh_partitioner']),
                  'reference_volume': reference_volume, 'sum_reference_weights': float(w.sum()),
                  'weight_definition': 'assembled scalar Quadrature test basis against reference dx, degree 2',
                  'wall_definition': 'mesh ellipsoidal/endo_dist, compared with saved norm_dist_endo; anatomical normalization must be reviewed',
                  'points': points}
        write_report(args.output, report)
        return 0 if report['passed'] else 1
    return root_call(comm, finish)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-json')
    parser.add_argument('--saved-output')
    parser.add_argument('--output')
    parser.add_argument('--compare', nargs=2)
    parser.add_argument('--seal-suite', help='write completion manifest after all reports pass')
    parser.add_argument('--image')
    parser.add_argument('--repo')
    arguments = parser.parse_args()
    if not arguments.seal_suite and not arguments.output:
        parser.error('--output is required for geometry export/comparison')
    if arguments.seal_suite:
        if not all((arguments.input_json, arguments.image, arguments.repo)):
            parser.error('--seal-suite requires --input-json, --image, --repo')
        seal_suite(arguments.seal_suite, arguments.input_json, arguments.image, arguments.repo)
        status = 0
    elif arguments.compare:
        status = compare(arguments.compare[0], arguments.compare[1], arguments.output)
    else:
        if not arguments.input_json:
            parser.error('--input-json is required')
        status = run(arguments)
    raise SystemExit(status)
