#!/usr/bin/env python
"""Validate a completed stride-one buffered-output smoke run (Python 2/3)."""
from __future__ import print_function

import argparse
import csv
import json
import os
import sys

import numpy as np


REQUIRED_SPATIAL = (
    'Ca_cytosol', 'Ca_SR', 'J_release', 'J_uptake',
    'k_act', 'k_serca', 'k_1', 'k_on', 'active_stress',
    'Sff_mesh', 'myofiber_passive', 'total_passive', 'hs_length',
    'fiber_strain', 'fiber_stretch_raw',
    'ecm_instantaneous_fiber_stress',
    'ecm_cycle_averaged_fiber_stress',
    'ecm_homeostatic_fiber_stress')


def count_rows_and_header(path):
    with open(path, 'r') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--expected-rows', type=int, required=True)
    parser.add_argument('--dt', type=float, default=0.001)
    args = parser.parse_args()

    report_path = os.path.join(args.output_dir, 'output_metadata.json')
    if not os.path.isfile(report_path):
        raise RuntimeError('missing %s' % report_path)
    report = json.load(open(report_path))
    failures = []
    if not report.get('complete', False):
        failures.append('output_metadata.json is not complete')
    for key in ('global_rows_written', 'spatial_rows_written'):
        if int(report.get(key, -1)) != args.expected_rows:
            failures.append('%s=%s, expected %d' %
                            (key, report.get(key), args.expected_rows))

    selected = set(report.get('selected_spatial_fields', []))
    for field in REQUIRED_SPATIAL:
        if field not in selected:
            failures.append('required spatial field not selected: %s' % field)
            continue
        path = os.path.join(args.output_dir, field + '_data.csv')
        if not os.path.isfile(path):
            failures.append('missing %s' % path)
            continue
        rows, header = count_rows_and_header(path)
        if rows != args.expected_rows:
            failures.append('%s has %d rows, expected %d' %
                            (field, rows, args.expected_rows))
        expected_points = len(header) - 2  # legacy index plus time
        if expected_points <= 0:
            failures.append('%s has no quadrature columns' % field)

    dof_path = os.path.join(args.output_dir, 'quadrature_global_dof.npy')
    coord_path = os.path.join(args.output_dir, 'quadrature_dof.npy')
    if not os.path.isfile(dof_path) or not os.path.isfile(coord_path):
        failures.append('missing quadrature DOF/coordinate metadata')
    else:
        dofs = np.load(dof_path)
        coords = np.load(coord_path)
        if coords.shape != (len(dofs), 3):
            failures.append('quadrature coordinate shape does not match DOFs')
        if not np.array_equal(dofs, np.arange(len(dofs))):
            failures.append('global quadrature DOFs are not canonical')

    data_path = os.path.join(args.output_dir, 'data.csv')
    try:
        # Stream the potentially very wide all-output global CSV. Loading
        # 60,001 rows and thousands of per-point baroreflex controls with
        # genfromtxt would defeat the bounded-memory output design.
        times = []
        with open(data_path, 'r') as handle:
            reader = csv.reader(handle)
            header = next(reader)
            time_index = header.index('time')
            for row in reader:
                times.append(float(row[time_index]))
        times = np.asarray(times, dtype=float)
        if len(times) != args.expected_rows:
            failures.append('data.csv has %d rows, expected %d' %
                            (len(times), args.expected_rows))
        if not np.all(np.isfinite(times)):
            failures.append('global timestamps are missing/non-finite')
        elif len(times) > 1 and not np.allclose(np.diff(times), args.dt,
                                                rtol=0.0, atol=1e-12):
            failures.append('global timestamp stride is not %.12g s' % args.dt)
    except Exception as error:
        failures.append('could not validate timestamps: %s' % error)

    result = {'passed': not failures, 'failures': failures,
              'expected_rows': args.expected_rows,
              'selected_spatial_field_count': len(selected)}
    out = os.path.join(args.output_dir, 'stride1_validation.json')
    with open(out, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
