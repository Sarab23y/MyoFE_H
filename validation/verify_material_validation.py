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

    # Central difference along F=diag(lambda,1,1/lambda). Because this path is
    # isochoric and sigma_nn=0, dW/dlambda=lambda*S_ff.
    value = 1.12
    delta = 1.0e-5
    minus = evaluate_state(biaxial_F(value-delta, 1.0), passive)
    centre = evaluate_state(biaxial_F(value, 1.0), passive)
    plus = evaluate_state(biaxial_F(value+delta, 1.0), passive)
    derivative = (plus['energy_total']-minus['energy_total'])/(2.0*delta)
    close('biaxial energy/PK2 derivative', derivative/value,
          centre['S_ff_total'])

    # For F=I+gamma e_f (x) e_s, dW/dgamma=P_fs=S_fs+gamma*S_ss.
    gamma = 0.15
    minus = evaluate_state(shear_F(gamma-delta, 'fiber_along_sheet'), passive)
    centre = evaluate_state(shear_F(gamma, 'fiber_along_sheet'), passive)
    plus = evaluate_state(shear_F(gamma+delta, 'fiber_along_sheet'), passive)
    derivative = (plus['energy_total']-minus['energy_total'])/(2.0*delta)
    expected = centre['S_fs_total'] + gamma*centre['S_ss_total']
    close('shear energy/PK2 derivative', derivative, expected)

    for row in (reference, centre):
        close('J', row['J'], 1.0)
        close('normal traction', row['traction_normal_residual'], 0.0)
        close('energy decomposition', row['energy_total'],
              row['energy_ground'] + row['energy_myofiber'] +
              row['energy_collagen'])
    if centre['energy_ground'] <= 0.0:
        raise AssertionError('Ground matrix did not activate in shear')
    if centre['energy_collagen'] <= 0.0:
        raise AssertionError('Directional collagen did not activate in shear')
    print('Material validation checks: PASS')


if __name__ == '__main__':
    main()
