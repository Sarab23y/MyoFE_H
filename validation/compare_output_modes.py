#!/usr/bin/env python
"""Compare short legacy/buffered runs without importing DOLFIN."""
from __future__ import division, print_function
import argparse
import csv
import json
import os
import sys
import numpy as np

GLOBAL_FIELDS = ('time', 'pressure_ventricle', 'volume_ventricle', 'heart_rate')
SPATIAL_FIELDS = ('active_stress', 'total_passive', 'Sff_mesh', 'hs_length',
                  'fiber_strain', 'ecm_instantaneous_fiber_stress',
                  'ecm_cycle_averaged_fiber_stress',
                  'ecm_homeostatic_fiber_stress')


def read_csv(path):
    mode = 'rb' if sys.version_info[0] < 3 else 'r'
    kwargs = {} if sys.version_info[0] < 3 else {'newline': ''}
    with open(path, mode, **kwargs) as handle:
        header = next(csv.reader(handle))
    values = np.genfromtxt(path, delimiter=',', skip_header=1)
    if values.ndim == 1:
        values = values.reshape((1, -1))
    return header, values


def maximum_difference(first, second):
    return 0.0 if first.size == 0 else float(np.max(np.abs(first-second)))


def comparison_block(first, second, tolerance):
    finite = bool(np.all(np.isfinite(first)) and np.all(np.isfinite(second)))
    difference = maximum_difference(first, second) if finite else None
    return {'passed': bool(finite and difference <= tolerance),
            'finite': finite, 'maximum_absolute_difference': difference,
            'tolerance': tolerance}


def compare_csv(first, second, fields, absolute_tolerance):
    h1, a = read_csv(first); h2, b = read_csv(second)
    missing = [name for name in fields if name not in h1 or name not in h2]
    same_shape = a.shape == b.shape; same_header = h1 == h2
    result = {'first_rows': int(a.shape[0]), 'second_rows': int(b.shape[0]),
              'missing_fields': missing, 'same_header': same_header,
              'same_shape': same_shape, 'fields': {}}
    passed = not missing and same_shape and same_header
    result['index'] = (comparison_block(a[:, 0], b[:, 0], 0.0)
                       if same_shape else {'passed': False})
    passed = passed and result['index']['passed']
    for name in fields:
        if name in missing or not same_shape:
            continue
        block = comparison_block(a[:, h1.index(name)],
                                 b[:, h2.index(name)], absolute_tolerance)
        result['fields'][name] = block; passed = passed and block['passed']
    result['passed'] = bool(passed)
    return result


def spatial_tolerance(field):
    # Legacy output rounds most stresses/HSL to one decimal and strain to five
    # decimal places. Thresholds sit just above half the final decimal unit.
    return 5.1e-6 if field == 'fiber_strain' else 5.1e-2


def compare_spatial(first_dir, second_dir, field, expected_dofs=None):
    first = os.path.join(first_dir, field + '_data.csv')
    second = os.path.join(second_dir, field + '_data.csv')
    if not os.path.isfile(first) or not os.path.isfile(second):
        return {'passed': False, 'missing': True}
    h1, a = read_csv(first); h2, b = read_csv(second)
    same_header = h1 == h2; same_shape = a.shape == b.shape
    layout_valid = len(h1) >= 3 and h1.count('time') == 1 and h1[-1] == 'time'
    point_headers = h1[1:-1] if layout_valid else []
    second_point_headers = h2[1:-1] if (len(h2) >= 3 and
        h2.count('time') == 1 and h2[-1] == 'time') else []
    expected_headers = ([str(int(value)) for value in expected_dofs]
                        if expected_dofs is not None else point_headers)
    first_dof_order_matches = point_headers == expected_headers
    second_dof_order_matches = second_point_headers == expected_headers
    dof_order_matches = first_dof_order_matches and second_dof_order_matches
    result = {'missing': False, 'same_header': same_header,
              'same_shape': same_shape, 'layout_valid': layout_valid,
              'quadrature_dof_order_matches': dof_order_matches,
              'first_quadrature_dof_order_matches': first_dof_order_matches,
              'second_quadrature_dof_order_matches': second_dof_order_matches,
              'first_shape': list(a.shape), 'second_shape': list(b.shape)}
    if not same_shape or not layout_valid:
        result.update({'passed': False, 'index': {'passed': False},
                       'timestamp': {'passed': False},
                       'gauss_points': {'passed': False}})
        return result
    result['index'] = comparison_block(a[:, 0], b[:, 0], 0.0)
    result['timestamp'] = comparison_block(a[:, -1], b[:, -1], 1.e-12)
    result['gauss_points'] = comparison_block(
        a[:, 1:-1], b[:, 1:-1], spatial_tolerance(field))
    # Backward-compatible keys now explicitly describe physical columns only.
    result['maximum_absolute_difference'] = result['gauss_points'][
        'maximum_absolute_difference']
    result['tolerance_for_legacy_rounding'] = spatial_tolerance(field)
    result['passed'] = bool(same_header and dof_order_matches and
        result['index']['passed'] and result['timestamp']['passed'] and
        result['gauss_points']['passed'])
    return result


def compare_mapping(first_dir, second_dir):
    report = {}; arrays = {}
    for filename in ('quadrature_dof.npy', 'quadrature_global_dof.npy'):
        paths = [os.path.join(directory, filename) for directory in
                 (first_dir, second_dir)]
        if not all(os.path.isfile(path) for path in paths):
            report[filename] = {'passed': False, 'missing': True}; continue
        a, b = np.load(paths[0]), np.load(paths[1])
        finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        equal = bool(a.shape == b.shape and finite and np.array_equal(a, b))
        report[filename] = {'passed': equal, 'missing': False,
                            'finite': finite, 'shape': list(a.shape)}
        if equal:
            arrays[filename] = a
    return report, arrays


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('legacy_output'); parser.add_argument('buffered_output')
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    mapping, arrays = compare_mapping(args.legacy_output, args.buffered_output)
    expected_dofs = arrays.get('quadrature_global_dof.npy')
    report = {'global': compare_csv(
        os.path.join(args.legacy_output, 'data.csv'),
        os.path.join(args.buffered_output, 'data.csv'), GLOBAL_FIELDS, 1.e-10),
        'spatial': {}, 'mapping': mapping}
    for name in SPATIAL_FIELDS:
        report['spatial'][name] = compare_spatial(
            args.legacy_output, args.buffered_output, name, expected_dofs)
    checks = [report['global']['passed']]
    checks += [item['passed'] for item in report['spatial'].values()]
    checks += [item['passed'] for item in report['mapping'].values()]
    report['required_spatial_fields'] = list(SPATIAL_FIELDS)
    report['all_required_spatial_fields_compared'] = bool(all(
        not item.get('missing', False) for item in report['spatial'].values()))
    checks.append(report['all_required_spatial_fields_compared'])
    report['passed'] = bool(all(checks))
    with open(args.report, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
