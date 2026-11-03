#!/usr/bin/env python
"""Compare converged passive-inflation volumes at matching pressures."""
from __future__ import print_function

import argparse

import numpy as np
import pandas as pd


def converged(path):
    data = pd.read_csv(path)
    mask = data['converged'].astype(str).str.lower() == 'true'
    return data.loc[mask, ['pressure_mmHg', 'cavity_volume_model_units']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('serial_csv')
    parser.add_argument('mpi_csv')
    parser.add_argument('--rtol', type=float, default=1.0e-8)
    parser.add_argument('--atol', type=float, default=1.0e-10)
    args = parser.parse_args()
    serial = converged(args.serial_csv)
    parallel = converged(args.mpi_csv)
    merged = serial.merge(parallel, on='pressure_mmHg',
                          suffixes=('_serial', '_mpi'))
    if merged.empty:
        raise ValueError('No matching converged pressures to compare')
    difference = (merged['cavity_volume_model_units_serial'] -
                  merged['cavity_volume_model_units_mpi'])
    merged['absolute_difference'] = np.abs(difference)
    print(merged.to_string(index=False))
    if not np.allclose(merged['cavity_volume_model_units_serial'],
                       merged['cavity_volume_model_units_mpi'],
                       rtol=args.rtol, atol=args.atol):
        raise AssertionError('Serial and MPI cavity volumes differ')
    print('Serial/MPI converged-volume comparison: PASS')


if __name__ == '__main__':
    main()
