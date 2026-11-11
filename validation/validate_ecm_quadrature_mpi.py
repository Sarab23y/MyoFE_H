#!/usr/bin/env python
"""FEniCS-2017.1/Python-2.7 validation of direct Quadrature evaluation.

This deliberately reports, rather than asserts away, the difference between
the historical DG1-to-Quadrature transfer and direct Quadrature projection.
"""
from __future__ import division, print_function
import argparse
import json
import math
import os
import sys

try:
    from .mpi_compat import as_mpi4py
except ImportError:
    from mpi_compat import as_mpi4py


def compare_reports(first_path, second_path, output_path):
    first = json.load(open(first_path)); second = json.load(open(second_path))
    first_keys = [tuple(item['coordinate']) for item in first['points']]
    second_keys = [tuple(item['coordinate']) for item in second['points']]
    a = dict((tuple(item['coordinate']), item['direct_fields']) for item in first['points'])
    b = dict((tuple(item['coordinate']), item['direct_fields']) for item in second['points'])
    missing = sorted(set(a).symmetric_difference(set(b)))
    duplicates = sorted(set([key for key in first_keys
                             if first_keys.count(key) > 1] +
                            [key for key in second_keys
                             if second_keys.count(key) > 1]))
    difference = max([abs(a[key][field]-b[key][field])
        for key in set(a).intersection(b) for field in a[key]] or [0.0])
    bad_coordinates = sorted(set(missing + duplicates))
    report = {'mode': 'cross_mpi_comparison', 'missing_or_duplicate_coordinates':
              [list(key) for key in bad_coordinates],
              'maximum_direct_value_difference': difference,
              'passed': not bad_coordinates and difference <= 1.0e-10}
    with open(output_path, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return 0 if report['passed'] else 1


def run(output_path):
    import numpy as np
    from dolfin import (Constant, FiniteElement, Function, FunctionSpace,
                        Identity, MixedElement, SpatialCoordinate,
                        UnitCubeMesh, VectorElement, VectorFunctionSpace,
                        as_tensor, conditional, det, diff, dx, exp, inner,
                        grad, interpolate, inv, project, split, sqrt, tr,
                        variable, assemble)
    mesh = UnitCubeMesh(2, 2, 2)
    cell = mesh.ufl_cell()
    qelement = FiniteElement('Quadrature', cell, degree=2,
                             quad_scheme='default')
    qspace = FunctionSpace(mesh, qelement)
    dg1 = FunctionSpace(mesh, 'DG', 1)
    x = SpatialCoordinate(mesh)
    # A deterministic positive-definite C field and the production-like
    # ground-plus-three-family collagen definition.
    C = variable(as_tensor(((1.10 + 0.02*x[0], 0.01*x[1], 0.0),
                            (0.01*x[1], 1.04 + 0.01*x[2], 0.0),
                            (0.0, 0.0, 1.02 + 0.01*x[0]))))
    f0 = Constant((1.0, 0.0, 0.0)); s0 = Constant((0.0, 1.0, 0.0)); n0 = Constant((0.0, 0.0, 1.0))
    ag, bg, ac, bc = 120.0, 5.0, 80.0, 4.0
    ground = ag/(2.0*bg)*(exp(bg*(tr(C)-3.0))-1.0)
    collagen = 0
    for direction in (f0, s0, n0):
        i4 = inner(direction, C*direction)
        i4star = conditional(i4 > 1.0, i4, 1.0)
        collagen += ac/(2.0*bc)*(exp(bc*(i4star-1.0)**2)-1.0)
    secm = 2.0*diff(0.274*ground + 0.026*collagen, C)
    expression = inner(f0, secm*f0)
    qparams = {'representation': 'quadrature', 'quadrature_degree': 2}
    direct = project(expression, qspace, form_compiler_parameters=qparams)
    pressure = Constant(12.0)
    diagnostic_expressions = {
        'ecm_fiber_stress': expression,
        'fiber_stretch_raw': sqrt(inner(f0, C*f0)),
        'fiber_strain': 0.5*(inner(f0, C*f0)-1.0),
        'J': sqrt(det(C)),
        'local_pressure': pressure,
        'incompressibility_fiber_stress': inner(
            f0, (-pressure*inv(C))*f0)}
    diagnostic_values = dict((name, project(value, qspace,
        form_compiler_parameters=qparams).vector().get_local())
        for name, value in diagnostic_expressions.items())
    # Reproduce the production mixed CG2-displacement/CG1-pressure coefficient
    # types. The pressure is deliberately extracted as a Function and
    # interpolated, avoiding the FFC-2017 mixed-coefficient Quadrature bug.
    mixed_space = FunctionSpace(mesh, MixedElement([
        VectorElement('CG', cell, 2), FiniteElement('CG', cell, 1)]))
    mixed_state = Function(mixed_space)
    displacement, unused_pressure = split(mixed_state)
    F_mixed = Identity(3) + grad(displacement)
    C_mixed = F_mixed.T*F_mixed
    mixed_expressions = {
        'mixed_fiber_stretch_raw': sqrt(inner(f0, C_mixed*f0)),
        'mixed_J': det(F_mixed),
        'mixed_inverse_C_fiber_factor': inner(f0, inv(C_mixed)*f0)}
    for name, value in mixed_expressions.items():
        diagnostic_values[name] = project(value, qspace,
            form_compiler_parameters=qparams).vector().get_local()
    diagnostic_values['mixed_pressure_safe'] = interpolate(
        mixed_state.sub(1, deepcopy=True), qspace).vector().get_local()
    historical = interpolate(project(expression, dg1,
        form_compiler_parameters={'representation': 'uflacs'}), qspace)
    difference = direct-historical
    numerator = float(assemble(difference*difference*dx(domain=mesh,
        metadata={'quadrature_degree': 2, 'quadrature_scheme': 'default'}),
        form_compiler_parameters=qparams))
    denominator = float(assemble(direct*direct*dx(domain=mesh,
        metadata={'quadrature_degree': 2, 'quadrature_scheme': 'default'}),
        form_compiler_parameters=qparams))
    values = direct.vector().get_local(); old = historical.vector().get_local()
    coordinates = qspace.tabulate_dof_coordinates().reshape(
        (-1, mesh.geometry().dim()))[:len(values)]
    point_metadata = {}
    try:
        global_cells = mesh.topology().global_indices(mesh.topology().dim())
    except Exception:
        global_cells = None
    for local_cell in range(mesh.num_cells()):
        for quadrature_point, local_dof in enumerate(
                qspace.dofmap().cell_dofs(local_cell)):
            local_dof = int(local_dof)
            if local_dof < len(values) and local_dof not in point_metadata:
                point_metadata[local_dof] = {
                    'local_cell': int(local_cell),
                    'global_cell': (int(global_cells[local_cell])
                                    if global_cells is not None else None),
                    'quadrature_point_in_cell': int(quadrature_point)}
    local = [{'coordinate': [round(float(v), 14) for v in coordinates[index]],
              'element': point_metadata.get(index, {}),
              'direct_ecm': float(values[index]),
              'direct_fields': dict((name, float(field_values[index]))
                                    for name, field_values in
                                    diagnostic_values.items()),
              'historical_dg1_transfer': float(old[index])}
             for index in range(len(values))]
    comm = as_mpi4py(mesh.mpi_comm())
    gathered = comm.gather(local, root=0)
    if comm.Get_rank() != 0:
        return 0
    points = [item for rank_items in gathered for item in rank_items]
    keys = [tuple(item['coordinate']) for item in points]
    finite = all(all(np.isfinite(value) for value in
                     item['direct_fields'].values()) and
                 np.isfinite(item['historical_dg1_transfer']) for item in points)
    report = {'mode': 'fenics_quadrature_validation', 'mpi_size': comm.Get_size(),
              'point_count': len(points), 'unique_coordinate_count': len(set(keys)),
              'finite': finite, 'maximum_absolute_dg1_direct_difference':
              max([abs(item['direct_ecm']-item['historical_dg1_transfer']) for item in points] or [0.0]),
              'relative_weighted_l2_difference': math.sqrt(numerator/denominator) if denominator else 0.0,
              'points': sorted(points, key=lambda item: item['coordinate'])}
    report['passed'] = finite and len(points) == len(set(keys))
    with open(output_path, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return 0 if report['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--compare', nargs=2)
    args = parser.parse_args(argv)
    if args.compare:
        return compare_reports(args.compare[0], args.compare[1], args.output)
    return run(args.output)


if __name__ == '__main__':
    sys.exit(main())
