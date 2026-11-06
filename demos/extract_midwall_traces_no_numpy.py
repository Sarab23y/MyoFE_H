#!/usr/bin/env python3
"""Extract eight selected MyoFE points; Python 3.6+ standard library only.

Run on LCC: python3 extract_midwall_traces_no_numpy.py
Reads existing CSVs without changing them. Writes a small ZIP beside sim_output.
Default window contains the previously inspected three complete healthy cycles.
CSV columns are matched by global DOF labels, never by column offsets.
No interpolation, smoothing, unit conversion or strain conversion is performed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
import zipfile

BASE = '/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_healthy_quad_validation_optimized/sim_output'
IDS = [1477, 577, 4629, 4633, 2589, 3041, 1941, 1473]
FIELDS = ['active_stress', 'fiber_strain', 'hs_length', 'ecm_instantaneous_fiber_stress']
CYCLES = [[37.484, 38.316], [38.316, 39.147], [39.147, 39.978]]


def extract(source, destination, start, stop, ids=None):
    count = 0
    first = last = previous = None
    gaps = []
    cycle_counts = [0, 0, 0]
    with source.open('r', newline='', encoding='utf-8-sig') as src, destination.open('w', newline='') as dst:
        reader = csv.reader(src)
        header = next(reader, None)
        if not header:
            raise ValueError('{} is empty'.format(source.name))
        header = [h.strip() for h in header]
        if header.count('time') != 1:
            raise ValueError('{} must have exactly one time column'.format(source.name))
        ti = header.index('time')
        if ids is None:
            wanted = ['time', 'pressure_ventricle', 'volume_ventricle', 'flow_ventricle_to_aorta',
                      'flow_veins_to_ventricle', 'activation', 'new_beat', 'heart_rate']
            names = [name for name in wanted if name in header]
            columns = [header.index(name) for name in names]
        else:
            lookup = {}
            for col, name in enumerate(header):
                if col == ti or (col == 0 and (not name or name.startswith('Unnamed:') or name == 'index')):
                    continue
                try:
                    gid = int(name)
                except ValueError:
                    raise ValueError('{}: unrecognized spatial column {!r}'.format(source.name, name))
                if gid in lookup:
                    raise ValueError('{}: duplicate DOF column {}'.format(source.name, gid))
                lookup[gid] = col
            missing = [gid for gid in ids if gid not in lookup]
            if missing:
                raise ValueError('{}: missing global DOF labels {}'.format(source.name, missing))
            columns = [ti] + [lookup[gid] for gid in ids]
            names = ['time'] + [str(gid) for gid in ids]
        writer = csv.writer(dst)
        writer.writerow(names)
        for lineno, row in enumerate(reader, 2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError('{}: wrong column count at line {}'.format(source.name, lineno))
            t = float(row[ti])
            if not math.isfinite(t):
                raise ValueError('{}: nonfinite time at line {}'.format(source.name, lineno))
            if previous is not None and t <= previous:
                raise ValueError('{}: time is not strictly increasing at line {}; no rows silently removed'.format(source.name, lineno))
            previous = t
            if not start <= t <= stop:
                continue
            values = [row[j] for j in columns]
            if ids is not None and any(not math.isfinite(float(v)) for v in values):
                raise ValueError('{}: nonfinite selected value at line {}'.format(source.name, lineno))
            if first is None:
                first = t
            if last is not None:
                gaps.append(t-last)
            last = t
            writer.writerow(values)
            count += 1
            for k, (a, b) in enumerate(CYCLES):
                if a <= t < b:
                    cycle_counts[k] += 1
    if count < 2:
        raise ValueError('{}: fewer than two samples in requested window'.format(source.name))
    return {'filename': source.name, 'rows': count, 'first_time': first, 'last_time': last,
            'min_dt': min(gaps), 'max_dt': max(gaps), 'rows_in_previous_reference_cycles': cycle_counts,
            'exported_columns': names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-output', default=BASE)
    parser.add_argument('--start', type=float, default=37.45)
    parser.add_argument('--stop', type=float, default=40.0)
    args = parser.parse_args()
    base = Path(args.sim_output)
    if not base.is_dir():
        raise ValueError('Simulation directory not found: {}'.format(base))
    if not math.isfinite(args.start) or not math.isfinite(args.stop) or args.stop <= args.start:
        raise ValueError('Expected finite start < stop')
    required = [base / (field + '_data.csv') for field in FIELDS]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        available = [p.name for p in sorted(base.glob('*.csv'))]
        raise ValueError('Missing required files: {}\nAvailable CSV filenames: {}\nPaste this message so the exact field names can be checked; no substitute stress is used.'.format(', '.join(missing), ', '.join(available)))
    output = base.parent / 'midwall_last_three_cycles_data.zip'
    report = {'global_dof_ids': IDS, 'requested_window_seconds': [args.start, args.stop],
              'previous_healthy_reference_cycles_seconds': CYCLES,
              'cycle_note': 'These intervals come from the previous healthy-run diagnostic; verify against the same-run global signals. They are not auto-detected last beats.',
              'source_folder': str(base), 'fields': [], 'warnings': [],
              'processing': 'Original values and times; global DOF header lookup; no interpolation, smoothing or unit conversion.',
              'geometry_note': 'Selected candidates at wall depth approximately 0.34449 and z_fraction 0.523785; not exact midwall. DOFs 1477 and 1473 are close neighbors.'}
    with tempfile.TemporaryDirectory(prefix='midwall_extract_') as temp:
        work = Path(temp)
        for source in required:
            print('Reading {} ...'.format(source.name), flush=True)
            info = extract(source, work / source.name, args.start, args.stop, IDS)
            report['fields'].append(info)
            print('  kept {} rows, {:.6f} to {:.6f} s'.format(info['rows'], info['first_time'], info['last_time']), flush=True)
            if not all(info['rows_in_previous_reference_cycles']):
                report['warnings'].append('{} does not contain samples in all previous reference cycles'.format(source.name))
            if info['first_time'] > CYCLES[0][0] or info['last_time'] < CYCLES[-1][1]:
                report['warnings'].append('{} does not bracket the full previous reference window; no extrapolation is permitted'.format(source.name))
        for filename, selected_ids in [('fiber_stretch_raw_data.csv', IDS), ('data.csv', None)]:
            source = base / filename
            if source.is_file():
                print('Reading optional {} ...'.format(filename), flush=True)
                report['fields'].append(extract(source, work / filename, args.start, args.stop, selected_ids))
            else:
                report['warnings'].append('Optional {} absent'.format(filename))
        geom = base.parent / 'midwall_midventricle_selection_no_numpy' / 'selected_midwall_midventricle_points.csv'
        if geom.is_file():
            with geom.open(newline='') as f:
                records = list(csv.DictReader(f))
            if set(int(r['global_dof']) for r in records) != set(IDS):
                raise ValueError('Selection CSV IDs differ from this extractor; do not combine different selections')
            (work / geom.name).write_bytes(geom.read_bytes())
        stability = base / 'ecm_baseline_stability.json'
        if stability.is_file():
            (work / stability.name).write_bytes(stability.read_bytes())
        (work / 'extraction_report.json').write_text(json.dumps(report, indent=2))
        staged = work / 'bundle.zip'
        with zipfile.ZipFile(str(staged), 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(work.iterdir()):
                if file != staged:
                    archive.write(str(file), file.name)
        # Copy completed archive only; failure during CSV validation leaves no partial ZIP.
        output.write_bytes(staged.read_bytes())
    for warning in report['warnings']:
        print('NOTE:', warning)
    print('\nDONE. Upload this ZIP to the chat:')
    print(output)
    print('Size: {:.2f} MB'.format(output.stat().st_size / 1024.0**2))
    print('Simulation inputs and outputs were not changed.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        raise SystemExit('ERROR: {}'.format(exc))
