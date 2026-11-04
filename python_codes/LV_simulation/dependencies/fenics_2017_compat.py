"""Narrow FEniCS-2017 helpers for scalar Quadrature evaluation."""
from __future__ import print_function


QUADRATURE_FORM_COMPILER_PARAMETERS = {
    'representation': 'quadrature',
    'quadrature_degree': 2
}
_REPORTED_FIELDS = set()


def project_quadrature(expression, space, field_name, comm=None):
    """Project with the backend required by legacy Quadrature elements.

    FFC 2017's uflacs path can emit references to undeclared Quadrature basis
    tables (for example ``FE11_C0_Q4``).  Naming every call makes a remaining
    cluster-only compiler failure attributable without altering the form.
    """
    from dolfin import project
    rank = comm.Get_rank() if comm is not None else 0
    report = rank == 0 and field_name not in _REPORTED_FIELDS
    if report:
        print('[QuadratureProjection] begin: %s' % field_name)
    try:
        result = project(expression, space,
            form_compiler_parameters=QUADRATURE_FORM_COMPILER_PARAMETERS)
    except Exception as error:
        raise RuntimeError('Quadrature projection failed for %s: %s' %
                           (field_name, str(error)))
    if report:
        print('[QuadratureProjection] complete: %s' % field_name)
        _REPORTED_FIELDS.add(field_name)
    return result


def mixed_pressure_quadrature(mixed_function, space, field_name, comm=None):
    """Evaluate the CG1 pressure subfunction without compiling a mixed form."""
    from dolfin import interpolate
    rank = comm.Get_rank() if comm is not None else 0
    report = rank == 0 and field_name not in _REPORTED_FIELDS
    if report:
        print('[QuadratureInterpolation] begin: %s' % field_name)
    try:
        pressure = mixed_function.sub(1, deepcopy=True)
        result = interpolate(pressure, space)
    except Exception as error:
        raise RuntimeError('Mixed pressure interpolation failed for %s: %s' %
                           (field_name, str(error)))
    if report:
        print('[QuadratureInterpolation] complete: %s' % field_name)
        _REPORTED_FIELDS.add(field_name)
    return result


def fiber_stress_components(functions, space, comm=None, prefix='diagnostic'):
    """Evaluate fiber PK2 components without a mixed-pressure FFC form.

    The decomposition is exactly the production identity
    ``passive = Xi + ground + collagen - p*C^-1`` and
    ``total = passive + active``.  Only diagnostic arrays are returned.
    """
    import numpy as np
    from dolfin import inner, inv
    f0 = functions['f0']
    expressions = {
        'xi_passive_stress': functions['Sff'],
        'ground_matrix_stress': inner(
            f0, functions['ground_matrix_passive']*f0),
        'collagen_stress': inner(f0, functions['collagen_passive']*f0),
        'active_stress': inner(f0, functions['Pactive']*f0),
        'inverse_C_fiber_factor': inner(
            f0, inv(functions['Fmat'].T*functions['Fmat'])*f0)}
    values = dict((name, project_quadrature(
        expression, space, prefix + '_' + name, comm).vector().get_local())
        for name, expression in expressions.items())
    values['local_pressure'] = mixed_pressure_quadrature(
        functions['w'], space, prefix + '_local_pressure', comm
        ).vector().get_local()
    values['incompressibility_stress'] = -values['local_pressure'] * \
        values.pop('inverse_C_fiber_factor')
    values['total_passive_stress'] = (
        values['xi_passive_stress'] + values['ground_matrix_stress'] +
        values['collagen_stress'] + values['incompressibility_stress'])
    values['total_fiber_stress'] = (
        values['total_passive_stress'] + values['active_stress'])
    for name, value in values.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError('Non-finite values in %s_%s' % (prefix, name))
    return values
