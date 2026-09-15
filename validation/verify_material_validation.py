#!/usr/bin/env python
"""Focused energy/stress and constraint checks for material validation."""
from __future__ import print_function

import argparse

import numpy as np

from material_point import (biaxial_F, evaluate_state, load_material_input,
                            shear_F)


def close(name, observed, expected, rtol=2.0e-4, atol=1.0e-7):
    if not np.isclose(observed, expected, rtol=rtol, atol=atol):
        raise AssertionError('%s: observed %g, expected %g' %
                             (name, observed, expected))


def directional_derivative(passive, state_function, value, delta):
    """Compare dW/dq with P:dF/dq for an isochoric one-parameter path."""
    F_minus = state_function(value-delta)
    F_centre = state_function(value)
    F_plus = state_function(value+delta)
    minus = evaluate_state(F_minus, passive)
    centre = evaluate_state(F_centre, passive)
    plus = evaluate_state(F_plus, passive)
    energy_derivative = (plus['energy_total']-minus['energy_total'])/(2*delta)
    S = np.array([
        [centre['S_ff_total'], centre['S_fs_total'], centre['S_fn_total']],
        [centre['S_fs_total'], centre['S_ss_total'], centre['S_sn_total']],
        [centre['S_fn_total'], centre['S_sn_total'], centre['S_nn_total']]])
    P = np.dot(F_centre, S)
    dF = (F_plus-F_minus)/(2*delta)
    return centre, energy_derivative, float(np.sum(P*dF))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    unused_config, unused_path, passive, unused_resolved = \
        load_material_input(args.config)

    reference = evaluate_state(np.eye(3), passive)
    close('reference energy', reference['energy_total'], 0.0)
    close('reference normal traction',
          reference['traction_normal_residual'], 0.0)
    close('reference J', reference['J'], 1.0)

    value = 1.12
    delta = 1.0e-5
    biaxial_paths = {
        'equal_biaxial': lambda q: biaxial_F(q, q),
        'fiber_dominant': lambda q: biaxial_F(q, 1.0+0.5*(q-1.0)),
        'sheet_dominant': lambda q: biaxial_F(1.0+0.5*(q-1.0), q),
    }
    for name, path in biaxial_paths.items():
        centre, observed, expected = directional_derivative(
            passive, path, value, delta)
        close(name + ' energy/work derivative', observed, expected)
        close(name + ' full traction', centre['traction_full_magnitude'], 0.0)

    gamma = 0.15
    shear_directions = ('fiber_along_sheet', 'sheet_along_fiber',
                        'fiber_along_normal')
    shear_states = []
    for direction in shear_directions:
        path = lambda q, d=direction: shear_F(q, d)
        centre, observed, expected = directional_derivative(
            passive, path, gamma, delta)
        close(direction + ' energy/work derivative', observed, expected)
        close(direction + ' normal traction',
              centre['traction_normal_residual'], 0.0)
        shear_states.append(centre)

    for row in [reference] + shear_states:
        close('J', row['J'], 1.0)
        close('normal traction', row['traction_normal_residual'], 0.0)
        close('energy decomposition', row['energy_total'],
              row['energy_ground'] + row['energy_myofiber'] +
              row['energy_collagen'])
    if any(row['energy_ground'] <= 0.0 for row in shear_states):
        raise AssertionError('Ground matrix did not activate in shear')
    if any(row['energy_collagen'] <= 0.0 for row in shear_states):
        raise AssertionError('Directional collagen did not activate in shear')
    print('Material validation checks: PASS')


if __name__ == '__main__':
    main()
