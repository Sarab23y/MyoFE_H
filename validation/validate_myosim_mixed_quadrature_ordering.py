#!/usr/bin/env python
"""Verify production MyoSim packing against actual DOLFIN dofmaps.

This is diagnostic-only.  It builds the same scalar Quadrature element and
repeated-scalar MixedElement used by MeshClass, then compares the production
point-major index ``q * ncomponents + component`` with the parent mixed-space
DOF obtained from each subspace's cell dofmap.
"""
from __future__ import division, print_function

import argparse
import hashlib
import json
import os
import sys

if __package__:
    from .mpi_compat import as_mpi4py, world_comm
    from .diagnostic_io import write_report, root_call, open_hdf5_file
else:
    from mpi_compat import as_mpi4py, world_comm
    from diagnostic_io import write_report, root_call, open_hdf5_file


def file_sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def _marker(global_q_dof, component):
    # Exactly representable as a double for the problem sizes used here.
    return float((int(component) + 1) * 1000000000 + int(global_q_dof))


def _state_component_count(input_json, explicit):
    if explicit is not None:
        return int(explicit)
    if not input_json:
        raise ValueError("provide --input-json or --state-components")
    import numpy as np
    with open(input_json) as handle:
        instruction = json.load(handle)
    impl = instruction['model']['half_sarcomere']['myofilaments']['implementation']
    low, high, width = [float(impl[key][0]) for key in ('bin_min', 'bin_max', 'bin_width')]
    if not np.isfinite([low, high, width]).all() or width <= 0 or high < low:
        raise ValueError('Invalid MyoSim bin bounds/width')
    bins = len(np.arange(low, high + width, width))
    scheme = impl['kinetic_scheme'][0]
    # Match myofilaments.set_up_3/4_state_with_SRX exactly, without importing
    # membrane/kinetics dependencies or changing the production model.
    if scheme in ('3_state_with_SRX', '3_state_with_SRX_and_exp_detach'):
        return bins + 4
    if scheme in ('4_state_with_SRX', '4_state_with_SRX_and_exp_detach'):
        return 2 * bins + 4
    raise ValueError('Unsupported MyoSim scheme: ' + scheme)


def resolve_mesh_path(input_json, repo):
    with open(input_json) as handle:
        configured = json.load(handle)['mesh']['mesh_path'][0]
    path = configured if os.path.isabs(configured) else os.path.join(repo, 'python_codes', configured)
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        raise IOError('Production mesh not found relative to REPO/python_codes: ' + path)
    return path


def _production_mesh(input_json, repo, dolfin_objects):
    HDF5File, Mesh, mpi_comm_world = dolfin_objects
    mesh_path = resolve_mesh_path(input_json, repo)
    mesh = Mesh()
    handle = open_hdf5_file(HDF5File, mpi_comm_world(), mesh_path, "r")
    try:
        handle.read(mesh, "ellipsoidal", False)
    finally:
        handle.close()
    return mesh, mesh_path


def _local_parent_map(mesh, scalar_space, mixed_space, components):
    """Return component/scalar-local-DOF -> mixed-parent-local-DOF."""
    mapping = [dict() for unused in range(components)]
    conflicts = []
    for cell_index in range(mesh.num_cells()):
        scalar_dofs = list(scalar_space.dofmap().cell_dofs(cell_index))
        for component in range(components):
            parent_dofs = list(
                mixed_space.sub(component).dofmap().cell_dofs(cell_index))
            if len(parent_dofs) != len(scalar_dofs):
                conflicts.append({"cell": int(cell_index),
                                  "component": int(component),
                                  "scalar_count": len(scalar_dofs),
                                  "parent_count": len(parent_dofs)})
                continue
            for scalar_dof, parent_dof in zip(scalar_dofs, parent_dofs):
                scalar_dof = int(scalar_dof)
                parent_dof = int(parent_dof)
                previous = mapping[component].get(scalar_dof)
                if previous is not None and previous != parent_dof:
                    conflicts.append({"cell": int(cell_index),
                                      "component": int(component),
                                      "scalar_dof": scalar_dof,
                                      "first_parent_dof": previous,
                                      "second_parent_dof": parent_dof})
                mapping[component][scalar_dof] = parent_dof
    return mapping, conflicts


def run(args):
    import numpy as np
    from dolfin import (FiniteElement, Function, FunctionSpace, HDF5File,
                        Mesh, MixedElement, UnitCubeMesh,
                        parameters)
    import dolfin

    # MeshClass explicitly selects SCOTCH for production MPI mesh loading.
    # Match it before constructing the mesh; do not rely on site defaults.
    if as_mpi4py(world_comm(dolfin)).Get_size() > 1:
        parameters['mesh_partitioner'] = 'SCOTCH'

    components = _state_component_count(args.input_json,
                                        args.state_components)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.manufactured_mesh:
        mesh = UnitCubeMesh(args.nx, args.ny, args.nz)
        mesh_source = "UnitCubeMesh(%d, %d, %d)" % (
            args.nx, args.ny, args.nz)
    else:
        if not args.input_json:
            raise ValueError("--input-json is required unless "
                             "--manufactured-mesh is selected")
        mesh, mesh_source = _production_mesh(
            args.input_json, repo, (HDF5File, Mesh, lambda: world_comm(dolfin)))
    comm = as_mpi4py(mesh.mpi_comm())
    qelement = FiniteElement("Quadrature", mesh.ufl_cell(), degree=2,
                             quad_scheme="default")
    qelement._quad_scheme = "default"
    scalar_space = FunctionSpace(mesh, qelement)
    mixed_space = FunctionSpace(
        mesh, MixedElement(components * [qelement]))
    scalar_owned = len(Function(scalar_space).vector().get_local())
    production_local_points = 4 * int(mesh.num_cells())
    mixed_function = Function(mixed_space)
    mixed_owned = len(mixed_function.vector().get_local())
    maps, conflicts = _local_parent_map(
        mesh, scalar_space, mixed_space, components)
    scalar_dofmap = scalar_space.dofmap()

    # Expected value at an actual mixed parent DOF, keyed by the physical
    # scalar Quadrature DOF and component found through cell/subspace dofmaps.
    expected = np.zeros(mixed_owned)
    expected_seen = np.zeros(mixed_owned, dtype=np.int32)
    point_major_index = {}
    map_errors = []
    for component in range(components):
        for scalar_local, parent_local in maps[component].items():
            if scalar_local >= scalar_owned or parent_local >= mixed_owned:
                continue                    # unowned/ghost entry
            global_q = int(scalar_dofmap.local_to_global_index(scalar_local))
            value = _marker(global_q, component)
            expected[parent_local] = value
            expected_seen[parent_local] += 1
            production_index = scalar_local * components + component
            point_major_index[(component, scalar_local)] = production_index
            if production_index >= mixed_owned:
                map_errors.append({"kind": "production_index_out_of_range",
                                   "component": component,
                                   "scalar_local_dof": scalar_local,
                                   "production_index": production_index})

    # Forward: current production packing -> recover through real subspace map.
    production = np.zeros(mixed_owned)
    for (component, scalar_local), index in point_major_index.items():
        if index < mixed_owned:
            global_q = int(scalar_dofmap.local_to_global_index(scalar_local))
            production[index] = _marker(global_q, component)
    mixed_function.vector().set_local(production)
    mixed_function.vector().apply("insert")
    recovered = mixed_function.vector().get_local()
    forward_bad = np.where((expected_seen == 1) &
                           (recovered != expected))[0]

    # Inverse: assign via authoritative subspace parent map -> decode through
    # the current production point-major indexing.
    mixed_function.vector().set_local(expected)
    mixed_function.vector().apply("insert")
    authoritative = mixed_function.vector().get_local()
    inverse_bad = []
    for (component, scalar_local), index in point_major_index.items():
        if index >= mixed_owned:
            continue
        global_q = int(scalar_dofmap.local_to_global_index(scalar_local))
        wanted = _marker(global_q, component)
        if authoritative[index] != wanted:
            inverse_bad.append(index)

    duplicate_parent = np.where(expected_seen > 1)[0]
    missing_parent = np.where(expected_seen == 0)[0]
    local_report = {
        "rank": int(comm.Get_rank()),
        "scalar_owned_dofs": int(scalar_owned),
        "production_local_point_count_4_times_mesh_cells":
            int(production_local_points),
        "mixed_owned_dofs": int(mixed_owned),
        "expected_mixed_owned_dofs": int(scalar_owned * components),
        "component_count": int(components),
        "cell_map_conflicts": conflicts[:args.max_details],
        "map_errors": map_errors[:args.max_details],
        "duplicate_parent_dof_count": int(len(duplicate_parent)),
        "missing_parent_dof_count": int(len(missing_parent)),
        "forward_mismatch_count": int(len(forward_bad)),
        "inverse_mismatch_count": int(len(inverse_bad)),
        "forward_mismatch_parent_dofs": [int(v) for v in
                                         forward_bad[:args.max_details]],
        "inverse_mismatch_production_indices": [int(v) for v in
                                                 inverse_bad[:args.max_details]]}
    local_report["passed"] = bool(
        production_local_points == scalar_owned and
        mixed_owned == scalar_owned * components and
        not conflicts and not map_errors and
        not len(duplicate_parent) and not len(missing_parent) and
        not len(forward_bad) and not len(inverse_bad))

    gathered = comm.gather(local_report, root=0)
    def finish():
        report = {
            "test": "myosim_mixed_quadrature_state_ordering",
            "mpi_size": int(comm.Get_size()),
            "mesh_source": mesh_source,
            "quadrature_degree": 2,
            "input_sha256": file_sha256(args.input_json) if args.input_json else None,
            "mesh_sha256": file_sha256(mesh_source) if not args.manufactured_mesh else None,
            "mesh_partitioner": str(parameters['mesh_partitioner']),
            "state_component_count": int(components),
            "packing_under_test": "point_major: q_local * components + component",
            "ranks": gathered,
            "passed": all(item["passed"] for item in gathered)}
        write_report(args.output, report)
        print("MyoSim mixed-quadrature ordering: %s" %
              ("PASS" if report["passed"] else "FAIL"))
        print("Report: %s" % args.output)
        return 0 if report["passed"] else 1
    return root_call(comm, finish)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-json")
    parser.add_argument("--state-components", type=int)
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--manufactured-mesh", action="store_true",
                        help="use UnitCubeMesh instead of configured LV mesh")
    parser.add_argument("--max-details", type=int, default=20)
    args = parser.parse_args(argv)
    if args.state_components is not None and args.state_components < 1:
        parser.error("--state-components must be positive")
    try:
        return run(args)
    except Exception as error:
        # Root writes a useful machine-readable failure when initialization
        # itself fails; other MPI ranks still return failure.
        try:
            import dolfin
            rank = as_mpi4py(world_comm(dolfin)).Get_rank()
        except Exception:
            # This standalone command starts on the world communicator. If
            # DOLFIN initialization failed, use mpi4py only to select the
            # single initialization-error reporter, never for mesh gathers.
            try:
                from mpi4py import MPI
                rank = MPI.COMM_WORLD.Get_rank()
            except Exception:
                rank = None
        if rank == 0 and not os.path.exists(args.output):
            try:
                write_report(args.output, {"test": "myosim_mixed_quadrature_state_ordering",
                                          "passed": False, "initialization_error": repr(error)})
            except Exception as report_error:
                print('Could not write failure report: %s' % report_error)
        print("MyoSim mixed-quadrature ordering: ERROR: %s" % error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
