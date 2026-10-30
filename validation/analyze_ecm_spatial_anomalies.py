#!/usr/bin/env python
"""Stream large MyoFE spatial CSVs and report ECM/mechanical anomalies.

This utility is deliberately independent of DOLFIN.  It never assumes that a
requested file exists and records mapping uncertainties instead of silently
substituting one stress definition for another.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os


DEFAULT_FIELDS = (
    'ecm_instantaneous_fiber_stress',
    'ecm_cycle_averaged_fiber_stress',
    'ecm_homeostatic_fiber_stress',
    'ecm_signed_stress_deviation',
    'ecm_positive_excess_stress',
    'active_stress', 'bulk_passive', 'myofiber_passive', 'incomp_stress',
    'total_passive', 'total_stress_spatial', 'Sff', 'Sff_mesh',
    'fiber_strain', 'alpha_f', 'hs_length', 'dx', 'dy', 'dz',
    'lx', 'ly', 'lz')

DEFAULT_WINDOWS = ((35.0, 40.0), (39.0, 40.0), (40.0, 50.0),
                   (75.0, 80.0))


def _finite(value):
    return not (math.isnan(value) or math.isinf(value))


def parse_windows(values):
    if not values:
        return list(DEFAULT_WINDOWS)
    windows = []
    for value in values:
        parts = value.split(':')
        if len(parts) != 2:
            raise ValueError('Window must use START:STOP syntax: %s' % value)
        start, stop = map(float, parts)
        if not stop > start:
            raise ValueError('Window stop must exceed start: %s' % value)
        windows.append((start, stop))
    return windows


def resolve_output_directory(path):
    path = os.path.abspath(path)
    if os.path.isdir(os.path.join(path, 'sim_output')):
        return os.path.join(path, 'sim_output')
    return path


def field_path(directory, field):
    candidates = [field + '_data.csv']
    if field == 'total_stress_spatial':
        candidates.insert(0, 'total_stress_spatial_data.csv')
    for candidate in candidates:
        path = os.path.join(directory, candidate)
        if os.path.isfile(path):
            return path
    return None


def read_main_times(directory):
    path = os.path.join(directory, 'data.csv')
    if not os.path.isfile(path):
        return []
    with open(path, 'r') as handle:
        reader = csv.DictReader(handle)
        if 'time' not in (reader.fieldnames or []):
            return []
        return [float(row['time']) for row in reader if row.get('time')]


def inspect_header(path):
    with open(path, 'r') as handle:
        header = next(csv.reader(handle))
    point_columns = []
    for index, name in enumerate(header):
        try:
            point_columns.append((int(name), index))
        except (TypeError, ValueError):
            pass
    time_index = header.index('time') if 'time' in header else None
    return {'header': header, 'point_columns': point_columns,
            'time_index': time_index}


def _window_for_time(time_value, windows):
    for index, (start, stop) in enumerate(windows):
        if start <= time_value <= stop:
            return index
    return None


def stream_field(path, targets, windows, fallback_times):
    """Read one field at a time and retain only summaries/target histories."""
    info = inspect_header(path)
    point_pairs = info['point_columns']
    point_ids = [item[0] for item in point_pairs]
    point_indices = [item[1] for item in point_pairs]
    point_lookup = dict(point_pairs)
    histories = dict((str(target), []) for target in targets if target in point_lookup)
    summaries = [dict(count=0, sum=0.0, minimum=None, maximum=None)
                 for unused in windows]
    extreme_min = None
    extreme_max = None
    nonfinite = 0
    rows = 0
    with open(path, 'r') as handle:
        reader = csv.reader(handle)
        next(reader)
        for row_index, row in enumerate(reader):
            if not row:
                continue
            if info['time_index'] is not None:
                time_value = float(row[info['time_index']])
            elif row_index < len(fallback_times):
                time_value = fallback_times[row_index]
            else:
                time_value = float(row_index)
            rows += 1
            for target, history in histories.items():
                raw = row[point_lookup[int(target)]]
                history.append([time_value, float(raw) if raw else None])
            window_index = _window_for_time(time_value, windows)
            for point_id, column_index in zip(point_ids, point_indices):
                raw = row[column_index]
                if raw == '':
                    continue
                value = float(raw)
                if not _finite(value):
                    nonfinite += 1
                    continue
                item = {'time': time_value, 'global_quadrature_dof': point_id,
                        'value': value}
                if extreme_min is None or value < extreme_min['value']:
                    extreme_min = item
                if extreme_max is None or value > extreme_max['value']:
                    extreme_max = item
                if window_index is not None:
                    summary = summaries[window_index]
                    summary['count'] += 1
                    summary['sum'] += value
                    summary['minimum'] = value if summary['minimum'] is None \
                        else min(summary['minimum'], value)
                    summary['maximum'] = value if summary['maximum'] is None \
                        else max(summary['maximum'], value)
    for summary, bounds in zip(summaries, windows):
        summary['window'] = list(bounds)
        summary['mean'] = (summary['sum']/summary['count']
                           if summary['count'] else None)
        del summary['sum']
    return {'path': path, 'rows': rows, 'point_count': len(point_ids),
            'point_columns': point_ids, 'target_histories': histories,
            'window_summaries': summaries, 'minimum': extreme_min,
            'maximum': extreme_max, 'nonfinite_count': nonfinite,
            'has_explicit_time': info['time_index'] is not None}


def verify_mapping(results, targets):
    fields = list(results)
    if not fields:
        return {'verified': False, 'reason': 'no spatial CSV fields found'}
    reference = results[fields[0]]['point_columns']
    mismatched = [name for name in fields[1:]
                  if results[name]['point_columns'] != reference]
    missing_targets = [target for target in targets if target not in reference]
    coordinate_fields = [name for name in ('lx', 'ly', 'lz') if name in results]
    coordinates_constant = {}
    for field in coordinate_fields:
        for target, history in results[field]['target_histories'].items():
            values = [item[1] for item in history if item[1] is not None]
            coordinates_constant[field + ':' + target] = \
                bool(values) and max(values) == min(values)
    return {'verified': not mismatched and not missing_targets and
                        len(coordinate_fields) == 3 and
                        all(coordinates_constant.values()),
            'reference_field': fields[0],
            'mismatched_field_headers': mismatched,
            'missing_target_dofs': missing_targets,
            'coordinate_fields_present': coordinate_fields,
            'selected_coordinates_constant': coordinates_constant,
            'interpretation': ('A numeric CSV column is a global scalar '
                               'quadrature DOF only when output assembly '
                               'diagnostics and this header/coordinate check '
                               'both pass.')}


def write_selected_histories(path, results, targets):
    fields = [field for field in results
              if results[field]['target_histories']]
    with open(path, 'w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['field', 'global_quadrature_dof', 'time', 'value'])
        for field in fields:
            for target in targets:
                for time_value, value in results[field]['target_histories'].get(
                        str(target), []):
                    writer.writerow([field, target, time_value, value])


def make_plots(output_directory, results, targets):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'matplotlib unavailable; plots skipped'
    for target in targets:
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        groups = (('ecm_', 'bulk_passive', 'myofiber_passive', 'total_passive',
                   'incomp_stress', 'total_stress_spatial'),
                  ('active_stress', 'Sff', 'Sff_mesh'),
                  ('fiber_strain', 'alpha_f', 'hs_length'),
                  ('dx', 'dy', 'dz'))
        plotted = False
        for axis, names in zip(axes, groups):
            for field, result in results.items():
                if not any(field.startswith(name) for name in names):
                    continue
                history = result['target_histories'].get(str(target), [])
                if history:
                    axis.plot([item[0] for item in history],
                              [item[1] for item in history], label=field)
                    plotted = True
            axis.axvline(40.0, color='black', linestyle='--')
            axis.axvspan(40.0, 50.0, color='orange', alpha=0.12)
            axis.grid(True); axis.legend(loc='best')
        if plotted:
            axes[-1].set_xlabel('time (s)')
            fig.suptitle('Global quadrature DOF %d' % target)
            fig.tight_layout()
            fig.savefig(os.path.join(output_directory,
                                     'ecm_dof_%d.png' % target), dpi=160)
        plt.close(fig)
    return 'plots written'


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory')
    parser.add_argument('--output', default=None,
                        help='Report directory (default: analysis_ecm)')
    parser.add_argument('--dofs', nargs='+', type=int,
                        default=[1757, 1773, 2317, 2496])
    parser.add_argument('--fields', nargs='+', default=list(DEFAULT_FIELDS))
    parser.add_argument('--window', action='append', default=[])
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    source = resolve_output_directory(args.output_directory)
    destination = args.output or os.path.join(source, 'analysis_ecm')
    if not os.path.isdir(destination):
        os.makedirs(destination)
    windows = parse_windows(args.window)
    main_times = read_main_times(source)
    results, missing = {}, []
    for field in args.fields:
        path = field_path(source, field)
        if path is None:
            missing.append(field)
            continue
        print('Reading %s' % path)
        results[field] = stream_field(path, args.dofs, windows, main_times)
    mapping = verify_mapping(results, args.dofs)
    report = {'source_directory': source, 'windows': [list(w) for w in windows],
              'target_global_quadrature_dofs': args.dofs,
              'mapping': mapping, 'missing_fields': missing,
              'fields': dict((name, dict((key, value) for key, value in data.items()
                                         if key not in ('point_columns',
                                                        'target_histories')))
                             for name, data in results.items()),
              'limitations': [
                  'Physical FE L2 differences require the quadrature mesh and '
                  'weights; they cannot be reconstructed from CSV coordinates alone.',
                  'Absent ECM-only CSV fields are not replaced with Sff, bulk, or '
                  'total stress because those are different stress definitions.',
                  'XDMF fields are visualization projections and are not lossless '
                  'quadrature-point histories.']}
    stability_candidates = [os.path.join(source, 'ecm_baseline_stability.json'),
                            os.path.join(os.path.dirname(source),
                                         'ecm_baseline_stability.json')]
    for candidate in stability_candidates:
        if os.path.isfile(candidate):
            report['baseline_stability_report'] = candidate
            break
    report_path = os.path.join(destination, 'ecm_spatial_anomaly_report.json')
    with open(report_path, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    write_selected_histories(os.path.join(destination,
                                          'selected_dof_histories.csv'),
                             results, args.dofs)
    plot_status = 'plots disabled' if args.no_plots else \
        make_plots(destination, results, args.dofs)
    print('Report: %s' % report_path)
    print('Mapping verified: %s' % mapping['verified'])
    print('Missing fields: %s' % ', '.join(missing))
    print(plot_status)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
