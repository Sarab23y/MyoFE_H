#!/usr/bin/env python
"""Compare a short legacy/buffered run without importing DOLFIN."""
from __future__ import division, print_function
import argparse
import csv
import json
import os
import sys
import numpy as np


GLOBAL_FIELDS = ('time', 'pressure_ventricle', 'volume_ventricle',
                 'heart_rate')
SPATIAL_FIELDS = ('active_stress', 'total_passive', 'Sff_mesh',
                  'hs_length', 'fiber_strain',
                  'ecm_instantaneous_fiber_stress')


def read_csv(path):
    mode = 'rb' if sys.version_info[0] < 3 else 'r'
    kwargs = {} if sys.version_info[0] < 3 else {'newline': ''}
    with open(path, mode, **kwargs) as handle:
        header = next(csv.reader(handle))
    values = np.genfromtxt(path, delimiter=',', skip_header=1)
    if values.ndim == 1:
        values = values.reshape((1, -1))
    return header, values


def compare_csv(first, second, fields, absolute_tolerance):
    h1, a = read_csv(first); h2, b = read_csv(second)
    missing = [name for name in fields if name not in h1 or name not in h2]
    result = {'first_rows': int(a.shape[0]), 'second_rows': int(b.shape[0]),
              'missing_fields': missing, 'fields': {}}
    passed = not missing and a.shape[0] == b.shape[0]
    for name in fields:
        if name in missing:
            continue
        x, y = a[:, h1.index(name)], b[:, h2.index(name)]
        difference = float(np.max(np.abs(x-y))) if len(x) else 0.0
        result['fields'][name] = {'maximum_absolute_difference': difference,
                                  'tolerance': absolute_tolerance}
        passed = passed and difference <= absolute_tolerance
    result['passed'] = bool(passed)
    return result


def compare_spatial(first_dir, second_dir, field):
    first = os.path.join(first_dir, field + '_data.csv')
    second = os.path.join(second_dir, field + '_data.csv')
    if not os.path.isfile(first) or not os.path.isfile(second):
        return {'passed': False, 'missing': True}
    h1, a = read_csv(first); h2, b = read_csv(second)
    # Legacy output rounds most stresses/HSL to one decimal and strain to five.
    tolerance = 5.1e-2 if field != 'fiber_strain' else 5.1e-6
    same_header = h1 == h2
    difference = float(np.max(np.abs(a-b))) if a.shape == b.shape else None
    return {'passed': bool(same_header and difference is not None and
                           difference <= tolerance),
            'same_header': same_header, 'first_shape': list(a.shape),
            'second_shape': list(b.shape),
            'maximum_absolute_difference': difference,
            'tolerance_for_legacy_rounding': tolerance}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('legacy_output'); parser.add_argument('buffered_output')
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    report = {'global': compare_csv(
        os.path.join(args.legacy_output, 'data.csv'),
        os.path.join(args.buffered_output, 'data.csv'), GLOBAL_FIELDS, 1.e-10),
        'spatial': {}, 'mapping': {}}
    for name in SPATIAL_FIELDS:
        report['spatial'][name] = compare_spatial(
            args.legacy_output, args.buffered_output, name)
    for filename in ('quadrature_dof.npy', 'quadrature_global_dof.npy'):
        paths = [os.path.join(directory, filename) for directory in
                 (args.legacy_output, args.buffered_output)]
        if not all(os.path.isfile(path) for path in paths):
            report['mapping'][filename] = {'passed': False, 'missing': True}
        else:
            a, b = np.load(paths[0]), np.load(paths[1])
            report['mapping'][filename] = {
                'passed': bool(a.shape == b.shape and np.array_equal(a, b)),
                'shape': list(a.shape)}
    checks = [report['global']['passed']]
    checks += [item['passed'] for item in report['spatial'].values()]
    checks += [item['passed'] for item in report['mapping'].values()]
    report['passed'] = bool(all(checks))
    with open(args.report, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
