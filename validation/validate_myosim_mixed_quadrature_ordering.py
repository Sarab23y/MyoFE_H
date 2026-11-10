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
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lv_path = os.path.join(repo, "python_codes", "LV_simulation")
    if lv_path not in sys.path:
        sys.path.insert(0, lv_path)
    from half_sarcomere import half_sarcomere as hs_module
    with open(input_json) as handle:
        instruction = json.load(handle)
    hs = hs_module.half_sarcomere(
        instruction["model"]["half_sarcomere"])
    return len(hs.myof.y)


def _production_mesh(input_json, repo, dolfin_objects):
    HDF5File, Mesh, mpi_comm_world = dolfin_objects
    with open(input_json) as handle:
        instruction = json.load(handle)
    configured = instruction["mesh"]["mesh_path"][0]
    candidates = [configured,
                  os.path.join(os.getcwd(), configured),
                  os.path.join(repo, "python_codes", configured),
                  os.path.join(os.path.dirname(input_json), configured)]
    mesh_path = None
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            mesh_path = candidate
            break
    if mesh_path is None:
        raise IOError("configured production mesh was not found; checked: %s" %
                      candidates)
    mesh = Mesh()
    handle = HDF5File(mpi_comm_world(), mesh_path, "r")
    handle.read(mesh, "ellipsoidal", False)
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
    from dolfin import (FiniteElement, Function, FunctionSpace, HDF5File, MPI,
                        Mesh, MixedElement, UnitCubeMesh, mpi_comm_world,
                        parameters)

    # MeshClass explicitly selects SCOTCH for production MPI mesh loading.
    # Match it before constructing the mesh; do not rely on site defaults.
    if MPI.size(mpi_comm_world()) > 1:
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
            args.input_json, repo, (HDF5File, Mesh, mpi_comm_world))
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
        "rank": int(MPI.rank(mesh.mpi_comm())),
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

    gathered = mesh.mpi_comm().gather(local_report, root=0)
    if MPI.rank(mesh.mpi_comm()) != 0:
        return 0 if local_report["passed"] else 1
    report = {
        "test": "myosim_mixed_quadrature_state_ordering",
        "mpi_size": int(MPI.size(mesh.mpi_comm())),
        "mesh_source": mesh_source,
        "quadrature_degree": 2,
        "input_sha256": file_sha256(args.input_json) if args.input_json else None,
        "mesh_sha256": file_sha256(mesh_source) if not args.manufactured_mesh else None,
        "mesh_partitioner": str(parameters['mesh_partitioner']),
        "state_component_count": int(components),
        "packing_under_test": "point_major: q_local * components + component",
        "ranks": gathered,
        "passed": all(item["passed"] for item in gathered)}
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print("MyoSim mixed-quadrature ordering: %s" %
          ("PASS" if report["passed"] else "FAIL"))
    print("Report: %s" % args.output)
    return 0 if report["passed"] else 1


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
            from dolfin import MPI, mpi_comm_world
            rank = MPI.rank(mpi_comm_world())
        except Exception:
            rank = 0
        if rank == 0:
            with open(args.output, "w") as handle:
                json.dump({"test": "myosim_mixed_quadrature_state_ordering",
                           "passed": False,
                           "initialization_error": repr(error)},
                          handle, indent=2, sort_keys=True)
        print("MyoSim mixed-quadrature ordering: ERROR: %s" % error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
