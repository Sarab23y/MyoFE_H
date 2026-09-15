#!/usr/bin/env python
"""Quasi-static, pressure-controlled passive LV inflation."""
from __future__ import print_function

import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np
from dolfin import Constant, TestFunction, TrialFunction, derivative, dx, solve, split
from mpi4py import MPI


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_CODES = os.path.join(ROOT, 'python_codes')
if PYTHON_CODES not in sys.path:
    sys.path.insert(0, PYTHON_CODES)

from LV_simulation.LV_simulation import LV_simulation  # noqa: E402
from LV_simulation.dependencies.recode_dictionary import recode  # noqa: E402


MMHG_PER_INTERNAL_PRESSURE = 0.0075


def _absolute_from_config(config_path, path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(ROOT, path))


def _pressure_targets(config):
    if 'pressures_mmHg' in config:
        return [float(value) for value in config['pressures_mmHg']]
    start = float(config['pressure_start_mmHg'])
    stop = float(config['pressure_stop_mmHg'])
    increment = float(config['pressure_increment_mmHg'])
    if increment <= 0.0 or stop < start:
        raise ValueError('Pressure range requires stop >= start and increment > 0')
    targets = list(np.arange(start, stop + 0.5*increment, increment))
    if targets[-1] < stop:
        targets.append(stop)
    return targets


def _required_volume_scale(config):
    value = config.get('volume_scale_to_ml')
    if not isinstance(value, (int, float)) or value <= 0.0:
        raise ValueError(
            'Set a positive volume_scale_to_ml after verifying the selected '
            'mesh coordinate units; no default conversion is assumed')
    return float(value)


def _restore_solution(lv, state):
    """Restore every mixed FE unknown and synchronize owned/ghost entries."""
    vector = lv.mesh.model['functions']['w'].vector()
    vector[:] = state
    vector.apply('insert')


def _build_pressure_controlled_problem(lv):
    """Reuse production passive/active/pressure residual components.

    Ftotal_gr replaces the volume constraint with prescribed endocardial
    pressure but leaves the unused Real cavity multiplier in the mixed space.
    The final term constrains only that unused multiplier to zero.
    """
    params = lv.mesh.model['solver_params']
    w = params['w']
    W = w.function_space()
    tests = split(TestFunction(W))
    unknowns = split(w)
    mesh = lv.mesh.model['mesh']
    mesh_volume = Constant(1.0)*dx(domain=mesh)
    cavity_multiplier_pin = unknowns[2]*tests[2]*mesh_volume
    residual = params['Ftotal_gr'] + cavity_multiplier_pin
    jacobian = derivative(residual, w, TrialFunction(W))
    return residual, jacobian, params['boundary_conditions']


def _solve_pressure_state(lv, pressure_mmHg, solver_parameters):
    lv.mesh.model['functions']['Press'].P = \
        pressure_mmHg/MMHG_PER_INTERNAL_PRESSURE
    residual, jacobian, bcs = _build_pressure_controlled_problem(lv)
    solver_result = solve(
        residual == 0, lv.mesh.model['functions']['w'], bcs, J=jacobian,
        solver_parameters={'newton_solver': solver_parameters},
        form_compiler_parameters={'representation': 'uflacs'})
    iterations = ''
    if isinstance(solver_result, tuple) and len(solver_result) > 0:
        iterations = solver_result[0]
    return float(lv.mesh.model['uflforms'].LVcavityvol()), iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as handle:
        config = json.load(handle)
    volume_scale_to_ml = _required_volume_scale(config)
    input_path = _absolute_from_config(config_path, config['input_json'])
    with open(input_path, 'r') as handle:
        instruction = recode(json.load(handle))

    # The production loader resolves this path from python_codes/. Reproduce
    # that established convention without changing the source JSON.
    configured_mesh_path = instruction['mesh']['mesh_path'][0]
    if not os.path.isabs(configured_mesh_path):
        instruction['mesh']['mesh_path'][0] = os.path.normpath(
            os.path.join(PYTHON_CODES, configured_mesh_path))

    # Construct the production mesh and weak form, but never enter its cardiac
    # time loop. Optional evolving models remain frozen.
    lv = LV_simulation(MPI.COMM_WORLD, instruction)

    # Active stress is a symbolic function of cb_number_density. Zeroing this
    # field makes Pactive identically zero without changing passive parameters.
    cb_density = lv.mesh.model['functions']['cb_number_density']
    original_cb_density = cb_density.vector().get_local().copy()
    cb_density.vector()[:] = 0.0
    cb_density.vector().apply('insert')
    local_active_density = np.max(np.abs(cb_density.vector().get_local())) \
        if len(cb_density.vector().get_local()) else 0.0
    global_active_density = MPI.COMM_WORLD.allreduce(
        local_active_density, op=MPI.MAX)
    if global_active_density != 0.0:
        raise RuntimeError('Failed to disable cross-bridge active stress')

    output_dir = os.path.join(os.path.dirname(config_path),
                              config['output_directory'])
    result_path = os.path.join(output_dir, 'passive_inflation.csv')
    if (MPI.COMM_WORLD.Get_rank() == 0 and os.path.exists(result_path) and
            not config.get('overwrite', False)):
        output_conflict = ('Refusing to overwrite existing validation result: '
                           + result_path)
    else:
        output_conflict = None
    output_conflict = MPI.COMM_WORLD.bcast(output_conflict, root=0)
    if output_conflict:
        raise RuntimeError(output_conflict)
    if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    MPI.COMM_WORLD.Barrier()

    targets = _pressure_targets(config)
    min_increment = float(config['minimum_pressure_increment_mmHg'])
    solver_parameters = config.get('newton_solver', {
        'relative_tolerance': 1.0e-7,
        'absolute_tolerance': 1.0e-7,
        'maximum_iterations': 50,
    })
    rows = []
    converged_pressure = 0.0
    state = lv.mesh.model['functions']['w'].vector().copy()
    pending = list(targets)
    terminal_failure = None
    while pending:
        target = pending.pop(0)
        _restore_solution(lv, state)
        local_error = None
        try:
            volume, iterations = _solve_pressure_state(
                lv, target, solver_parameters)
        except RuntimeError as error:
            local_error = str(error)
        failed_ranks = MPI.COMM_WORLD.allreduce(
            1 if local_error is not None else 0, op=MPI.SUM)
        if failed_ranks == 0:
            state = lv.mesh.model['functions']['w'].vector().copy()
            converged_pressure = target
            rows.append({'pressure_mmHg': target,
                         'pressure_internal': target/MMHG_PER_INTERNAL_PRESSURE,
                         'cavity_volume_model_units': volume,
                         'cavity_volume_ml': volume*volume_scale_to_ml,
                         'newton_iterations': iterations,
                         'converged': True})
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('Converged: pressure=%g mmHg, volume=%g model units' %
                      (target, volume))
        else:
            messages = MPI.COMM_WORLD.allgather(local_error)
            message = next(item for item in messages if item is not None)
            failed = float(target - converged_pressure)
            rows.append({'pressure_mmHg': target,
                         'pressure_internal': target/MMHG_PER_INTERNAL_PRESSURE,
                         'cavity_volume_model_units': '',
                         'cavity_volume_ml': '',
                         'newton_iterations': '',
                         'converged': False,
                         'message': message})
            _restore_solution(lv, state)
            if failed/2.0 < min_increment:
                terminal_failure = (
                    'Passive inflation failed at %g mmHg; minimum increment '
                    '%g mmHg reached' % (target, min_increment))
                break
            midpoint = converged_pressure + failed/2.0
            pending.insert(0, target)
            pending.insert(0, midpoint)
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('Retrying with reduced step at %g mmHg' % midpoint)

    cb_density.vector()[:] = original_cb_density
    cb_density.vector().apply('insert')
    if MPI.COMM_WORLD.Get_rank() == 0:
        csv_path = result_path
        fields = ['pressure_mmHg', 'pressure_internal',
                  'cavity_volume_model_units', 'cavity_volume_ml',
                  'newton_iterations', 'converged', 'message']
        with open(csv_path, 'wb') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            'test': 'passive LV pressure-controlled inflation',
            'input_json': input_path,
            'configuration': config,
            'active_contraction': 'disabled by zero cb_number_density field',
            'growth_and_remodeling': 'not advanced; cardiac time loop bypassed',
            'intracellular_passive': ('FE Xi passive myofiber included; '
                                      'standalone 1D passive function not added'),
            'pressure_conversion': 'mmHg = 0.0075 * internal pressure',
            'volume_scale_to_ml': volume_scale_to_ml,
            'volume_units': ('mL column uses the explicit configured scale; '
                             'verify it for the selected mesh'),
            'reference_assumption': config['reference_assumption'],
        }
        with open(os.path.join(output_dir, 'passive_inflation_metadata.json'),
                  'w') as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        shutil.copyfile(
            input_path,
            os.path.join(output_dir, 'input_instruction_snapshot.json'))
        print('Wrote passive inflation results to %s' % output_dir)
    terminal_failure = MPI.COMM_WORLD.bcast(terminal_failure, root=0)
    if terminal_failure:
        raise RuntimeError(terminal_failure)


if __name__ == '__main__':
    main()
