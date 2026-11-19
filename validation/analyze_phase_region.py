#!/usr/bin/env python3
"""Read-only, two-pass phase/regional diagnostics (Python >=3.6, NumPy).

Spatial arrays are never retained over time. Saved timestamps are required;
no row-number time inference, interpolation, or stress substitution is used.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_ecm_spatial_anomalies import field_path, inspect_header, resolve_output_directory
    from .export_verified_quadrature_geometry import require_suite, REPORTS, SOURCES
else:
    from analyze_ecm_spatial_anomalies import field_path, inspect_header, resolve_output_directory
    from export_verified_quadrature_geometry import require_suite, REPORTS, SOURCES

FIELDS = ('hs_length', 'fiber_stretch_raw', 'fiber_strain', 'active_stress',
          'Sff_mesh', 'ecm_instantaneous_fiber_stress')
OPTIONAL = ('Ca_cytosol', 'Ca_SR', 'J_release', 'J_uptake')
PHASES = ('isovolumetric_contraction', 'ejection', 'isovolumetric_relaxation', 'filling')
GLOBAL = ('time', 'new_beat', 'activation', 'heart_rate', 'pressure_ventricle',
          'volume_ventricle', 'pressure_aorta', 'pressure_atrium', 'pressure_veins',
          'flow_ventricle_to_aorta', 'flow_veins_to_ventricle')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_global(path):
    data = {}
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('duplicate global headers')
        keys = [k for k in GLOBAL if k in reader.fieldnames]
        data = {k: [] for k in keys}
        for row in reader:
            for k in keys:
                data[k].append(float(row[k]))
    data = {k: np.asarray(v) for k, v in data.items()}
    for k in ('time', 'pressure_ventricle', 'volume_ventricle'):
        if k not in data:
            raise ValueError('missing required global signal: ' + k)
    if len(data['time']) < 3 or not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError('global signals are too short or nonfinite')
    if np.any(np.diff(data['time']) <= 0):
        raise ValueError('global time must be strictly increasing')
    return data


def phase_cycles(g, flow_threshold=1e-8, volume_rate_threshold=1e-5):
    """Estimate events without fixed timing; reject ambiguous/reopening cycles.

    Prefer forward inlet-flow closures as cycle boundaries. Fallback beat
    markers are explicitly estimates of MVC, not observed valve closures.
    Pressure/volume fallbacks carry lower-confidence provenance.
    """
    t, p, v = (g[k] for k in ('time', 'pressure_ventricle', 'volume_ventricle'))
    dv = np.gradient(v, t)
    if 'flow_ventricle_to_aorta' in g:
        outflow = g['flow_ventricle_to_aorta'] > flow_threshold
        out_method = 'forward_aortic_flow'
    elif 'pressure_aorta' in g:
        outflow = p > g['pressure_aorta']
        out_method = 'ventricular_aortic_pressure_difference_estimate'
    else:
        outflow = dv < -volume_rate_threshold
        out_method = 'volume_derivative_estimate'
    if 'flow_veins_to_ventricle' in g:
        inflow = g['flow_veins_to_ventricle'] > flow_threshold
        in_method = 'forward_inlet_flow'
    elif 'pressure_atrium' in g:
        inflow = g['pressure_atrium'] > p
        in_method = 'atrial_ventricular_pressure_difference_estimate'
    elif 'pressure_veins' in g:
        inflow = g['pressure_veins'] > p
        in_method = 'model_venous_inlet_ventricular_pressure_difference_estimate'
    else:
        inflow = dv > volume_rate_threshold
        in_method = 'volume_derivative_estimate'
    closures = np.flatnonzero(inflow[:-1] & ~inflow[1:]) + 1
    boundary_method = in_method + '_closure'
    if len(closures) < 2:
        if 'new_beat' not in g:
            raise ValueError('no complete inlet-closure cycles and no new_beat fallback')
        beat = g['new_beat'] > 0
        closures = np.flatnonzero(beat & ~np.r_[False, beat[:-1]])
        boundary_method = 'new_beat_MVC_estimate'
    cycles, rejected = [], []
    nominal_dt = float(np.median(np.diff(t)))
    for number, (a, b) in enumerate(zip(closures[:-1], closures[1:])):
        reason = None
        ao = np.flatnonzero(outflow[a:b]) + a
        if not len(ao):
            reason = 'no ejection'
        else:
            start = int(ao[0])
            stop = int(ao[-1] + 1)
            mo = np.flatnonzero(inflow[stop:b]) + stop
            if not len(mo):
                reason = 'no post-ejection filling'
            else:
                opening = int(mo[0])
                if not a < start < stop < opening < b:
                    reason = 'unresolved isovolumetric interval or event order'
                elif not np.all(outflow[start:stop]) or np.any(inflow[a:opening]) or not np.all(inflow[opening:b]):
                    reason = 'reopening, chatter, or overlapping valve intervals'
                elif np.any(np.diff(t[a:b+1]) > 1.5 * nominal_dt):
                    reason = 'global time gap'
        if reason:
            rejected.append({'candidate': number, 'start': float(t[a]), 'end': float(t[b]), 'reason': reason})
            continue
        peak = a + int(np.argmax(p[a:b]))
        cuts = [int(a), start, stop, opening, int(b)]
        reverse = {name: int(np.count_nonzero(g[name][a:b] < -flow_threshold))
                   for name in ('flow_ventricle_to_aorta', 'flow_veins_to_ventricle') if name in g}
        consistency = {'reverse_flow_samples': reverse,
                       'volume_change_IVC': float(v[start]-v[a]),
                       'volume_change_IVR': float(v[opening]-v[stop])}
        if 'pressure_aorta' in g:
            consistency['outflow_pressure_sign_disagreement_samples'] = int(np.count_nonzero(
                outflow[a:b] != (p[a:b] > g['pressure_aorta'][a:b])))
        cycles.append({'candidate': number, 'indices': cuts, 'times': [float(t[i]) for i in cuts],
                       'boundary_method': boundary_method, 'aortic_method': out_method,
                       'inlet_method': in_method, 'peak_pressure_time': float(t[peak]),
                       'peak_pressure': float(p[peak]), 'end_diastole_time': float(t[a]),
                       'end_systole_time': float(t[stop]), 'ed_volume': float(v[a]),
                       'es_volume': float(v[stop]), 'stroke_volume': float(v[a]-v[stop]),
                       'ejection_volume_change': float(v[start]-v[stop]),
                       'phase_durations': dict(zip(PHASES, np.diff(t[cuts]).tolist())),
                       'timing_uncertainty_s_at_least': float(max(np.diff(t[a:b+1]))),
                       'exact_valve_events_identified': False,
                       'coupling_stagger_uncertainty_s': None,
                       'signal_consistency': consistency,
                       'warning': 'Saved signals may be staggered; event times are sampled estimates. '
                                  'Flow is cached from circulation ODE evaluations, while saved ventricular '
                                  'pressure is updated after mechanics. The extra staging error is unquantified. '
                                  'Forward-flow cessation is not an independently measured valve state. '
                                  'Isovolumetric labels are nominal if volume changes or reverse flow occur.'})
    return cycles, rejected


def geometry(path, directory, hotspots, axis=2, tolerance=1e-9):
    """Require independent FEniCS verification and hash-bound saved metadata."""
    with open(path) as handle:
        report = json.load(handle)
    if report.get('passed') is not True or report.get('saved_mapping_verified') is not True:
        raise ValueError('geometry report must pass saved mapping verification')
    if set(report.get('source_sha256', {})) != {'quadrature_dof.npy', 'quadrature_global_dof.npy', 'norm_dist_endo.npy'}:
        raise ValueError('geometry report lacks required source hashes')
    for name, expected in report['source_sha256'].items():
        if digest(Path(directory) / name) != expected:
            raise ValueError('geometry source hash mismatch: ' + name)
    points = report['points']
    ids = [int(x['global_dof']) for x in points]
    xyz = np.asarray([x['coordinate'] for x in points], float)
    wall = np.asarray([x['wall_depth'] for x in points], float)
    if len(ids) != len(set(ids)) or xyz.shape != (len(ids), 3):
        raise ValueError('duplicate IDs or invalid coordinate shape')
    if not np.isfinite(xyz).all() or not np.isfinite(wall).all() or np.any((wall < 0) | (wall > 1)):
        raise ValueError('invalid coordinates or normalized wall depth')
    # Refuse coincident or effectively coincident coordinates rather than
    # silently counting duplicated physical locations as independent samples.
    seen = {}
    for gid, coord in zip(ids, xyz):
        key = tuple(np.floor(coord / tolerance).astype(np.int64))
        for offset in np.ndindex(3, 3, 3):
            near = tuple(key[j] + offset[j] - 1 for j in range(3))
            for other in seen.get(near, []):
                if np.linalg.norm(coord-other) <= tolerance:
                    raise ValueError('coincident quadrature coordinates')
        seen.setdefault(key, []).append(coord)
    if not set(hotspots).issubset(ids):
        raise ValueError('missing hotspot IDs')
    extent = np.ptp(xyz[:, axis])
    if extent <= 0:
        raise ValueError('long axis has zero extent')
    z = (xyz[:, axis] - xyz[:, axis].min()) / extent
    selected = np.flatnonzero((wall >= .3) & (wall <= .7) & (z >= .3) & (z <= .7))
    if not len(selected):
        raise ValueError('empty midwall/midventricular region')
    weights = None
    if report.get('weights_verified') is True:
        weights = np.asarray([x['reference_weight'] for x in points], float)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('invalid reference quadrature weights')
        volume = report.get('reference_volume')
        if volume is None or not np.isfinite(volume) or volume <= 0 or not np.isclose(weights.sum(), volume, rtol=1e-10, atol=1e-12):
            raise ValueError('quadrature weights do not sum to reference volume')
    # Recheck the saved arrays themselves, not only the report's pass flag.
    saved_ids = np.load(Path(directory)/'quadrature_global_dof.npy').reshape(-1)
    saved_xyz = np.load(Path(directory)/'quadrature_dof.npy')
    saved_wall = np.load(Path(directory)/'norm_dist_endo.npy').reshape(-1)
    if len(saved_ids) != len(ids) or len(set(saved_ids)) != len(ids) or set(saved_ids) != set(ids):
        raise ValueError('geometry report/saved point IDs differ')
    saved_index = {gid: i for i, gid in enumerate(saved_ids)}
    order = [saved_index[i] for i in ids]
    if saved_xyz.shape != xyz.shape or saved_wall.shape != wall.shape or not np.allclose(saved_xyz[order], xyz, rtol=0, atol=1e-10) or not np.allclose(saved_wall[order], wall, rtol=0, atol=1e-10):
        raise ValueError('geometry report/saved coordinate or wall tuples differ')
    for point in points:
        vertices = np.asarray(point.get('cell_reference_vertices'), float)
        q = point.get('quadrature_point_in_cell')
        if vertices.shape != (4, 3) or not np.isfinite(vertices).all() or q not in range(4):
            raise ValueError('missing/invalid cell or local quadrature metadata')
        matrix = (vertices[1:]-vertices[0]).T
        try:
            tail = np.linalg.solve(matrix, np.asarray(point['coordinate'])-vertices[0])
        except np.linalg.LinAlgError:
            raise ValueError('degenerate reference cell')
        barycentric = np.r_[1.-tail.sum(), tail]
        if np.any(barycentric < -1e-9) or np.any(barycentric > 1.+1e-9):
            raise ValueError('quadrature coordinate lies outside its reference cell')
    return report, ids, xyz, wall, z, selected, weights


def spatial_rows(path, expected_ids):
    info = inspect_header(str(path))
    pairs = info['point_columns']
    ids = [i for i, unused in pairs]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('spatial point identities differ: ' + str(path))
    if info['time_index'] is None or len(info['header']) != len(set(info['header'])):
        raise ValueError('missing time or duplicate headers: ' + str(path))
    lookup = dict(pairs)
    columns = [lookup[i] for i in expected_ids]
    previous = -float('inf')
    with open(path, newline='') as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) != len(info['header']):
                raise ValueError('incomplete spatial row: ' + str(path))
            t = float(row[info['time_index']])
            if not np.isfinite(t) or t <= previous:
                raise ValueError('nonfinite/duplicate/decreasing spatial time: ' + str(path))
            previous = t
            yield t, row, columns


def aligned_times(path, ids, times, tolerance=1e-9):
    """Map timestamps only, with strict tolerance; never interpolate states."""
    result = set()
    j = 0
    for t, unused, columns in spatial_rows(path, ids):
        while j < len(times) and times[j] < t-tolerance:
            j += 1
        if j == len(times):
            break
        if abs(times[j]-t) > tolerance:
            raise ValueError('spatial timestamp is off the global grid: ' + str(path))
        if j in result:
            raise ValueError('multiple spatial rows match one global timestamp')
        result.add(j)
    return result


def choose_cycles(cycles, common, count, start, end):
    if count < 5:
        raise ValueError('at least five cycles are required')
    run, latest = [], None
    for cycle in cycles:
        a, b = cycle['indices'][0], cycle['indices'][-1]
        valid = (start is None or cycle['times'][0] >= start) and cycle['times'][-1] <= end and all(i in common for i in range(a, b+1))
        if not valid:
            run = []
            continue
        if run and cycle['candidate'] != run[-1]['candidate']+1:
            run = []
        run.append(cycle)
        if len(run) >= count:
            if start is not None:
                return run[:count]
            latest = run[-count:]
    if latest is not None:
        return latest
    raise ValueError('not enough consecutive complete, aligned cycles in requested window')


def stats(values, weights=None):
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('missing/nonfinite regional values')
    result = dict(zip(('p05', 'p25', 'median', 'p75', 'p95'), np.percentile(a, [5, 25, 50, 75, 95]).tolist()))
    result.update(point_mean=float(a.mean()), point_std=float(a.std()), minimum=float(a.min()), maximum=float(a.max()))
    if weights is not None:
        mean = float(np.average(a, weights=weights))
        result.update(volume_weighted_mean=mean,
                      volume_weighted_std=float(np.sqrt(np.average((a-mean)**2, weights=weights))))
    return result


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def analyze(args):
    directory = Path(resolve_output_directory(str(args.data)))
    out = Path(args.output)
    if out.exists():
        raise ValueError('refusing to overwrite output directory')
    repo = Path(__file__).resolve().parents[1]
    submitted_input = args.input_json or directory.parent/'instruction.json'
    manifest = require_suite(str(args.mapping), str(submitted_input), str(repo))
    geometry_path = Path(args.mapping)/'geometry_32.json'
    report, ids, xyz, wall, z, region, weights = geometry(geometry_path, directory, args.hotspots, args.axis)
    rw = None if weights is None else weights[region]
    paths = {f: field_path(str(directory), f) for f in FIELDS + OPTIONAL}
    missing = [f for f in FIELDS if paths[f] is None]
    if missing:
        raise ValueError('missing required fields (no stress substitution): ' + ', '.join(missing))
    paths = {f: Path(p) for f, p in paths.items() if p}
    sources = ([directory / 'data.csv', submitted_input, Path(args.mapping)/'mapping_manifest.json'] +
               [Path(args.mapping)/name for name in REPORTS] + [repo/name for name in SOURCES] +
               list(paths.values()) + [directory / name for name in report['source_sha256']])
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sources}
    g = read_global(directory / 'data.csv')
    cycles, rejected = phase_cycles(g, args.flow_threshold, args.volume_rate_threshold)
    common = set(range(len(g['time'])))
    coverage = {}
    for field, path in paths.items():
        available = aligned_times(path, ids, g['time'], args.time_tolerance)
        coverage[field] = len(available)
        common.intersection_update(available)
    selected = choose_cycles(cycles, common, args.cycles, args.start, args.end)
    out.mkdir(parents=True)
    write_csv(out / 'region_geometry.csv', ({'global_dof': ids[i], 'x': xyz[i, 0], 'y': xyz[i, 1], 'z': xyz[i, 2],
              'wall_depth': wall[i], 'longitudinal_fraction': z[i],
              'reference_weight': '' if weights is None else weights[i]} for i in region))
    write_csv(out / 'global_signals.csv', (dict((k, float(v[i])) for k, v in g.items())
              for i in range(selected[0]['indices'][0], selected[-1]['indices'][-1]+1)))
    labels = {}
    for ci, cyc in enumerate(selected):
        for phase, a, b in zip(PHASES, cyc['indices'][:-1], cyc['indices'][1:]):
            for i in range(a, b):
                labels[i] = (ci+1, phase)
    summaries, plots, anomalies, point_comparisons = [], {}, [], {}
    hotspot_indices = [ids.index(i) for i in args.hotspots]
    for field, path in paths.items():
        regional_path = out / (field + '_region.csv')
        hotspot_path = out / (field + '_hotspots.csv')
        accum = {}
        series = []
        previous = None
        grid_index = 0
        with open(regional_path, 'w', newline='') as rh, open(hotspot_path, 'w', newline='') as hh:
            region_writer = hotspot_writer = None
            for t, raw, cols in spatial_rows(path, ids):
                while grid_index < len(g['time']) and g['time'][grid_index] < t-args.time_tolerance:
                    grid_index += 1
                if grid_index not in labels:
                    if t > selected[-1]['times'][-1]:
                        break
                    continue
                if abs(g['time'][grid_index]-t) > args.time_tolerance:
                    raise ValueError('time alignment changed between passes')
                values = np.asarray([float(raw[c]) for c in cols])
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite field values: ' + field)
                a = values[region]
                cycle, phase = labels[grid_index]
                row = {'time': t, 'cycle': cycle, 'phase': phase}
                row.update(stats(a, rw))
                row['minimum_global_dof'] = ids[region[int(np.argmin(a))]]
                row['maximum_global_dof'] = ids[region[int(np.argmax(a))]]
                if field == 'fiber_stretch_raw':
                    row['point_fraction_below_reference_length'] = float(np.mean(a < 1))
                    row['volume_fraction_below_reference_length'] = '' if rw is None else float(np.average(a < 1, weights=rw))
                    screen = (a < args.low_stretch) | (a > args.high_stretch)
                    row['region_point_fraction_outside_screen'] = float(screen.mean())
                    row['region_volume_fraction_outside_screen'] = '' if rw is None else float(np.average(screen, weights=rw))
                    row['whole_mesh_point_fraction_outside_screen'] = float(np.mean((values < args.low_stretch) | (values > args.high_stretch)))
                    row['whole_mesh_minimum'] = float(values.min())
                    row['whole_mesh_minimum_global_dof'] = ids[int(values.argmin())]
                    row['whole_mesh_maximum'] = float(values.max())
                    row['whole_mesh_maximum_global_dof'] = ids[int(values.argmax())]
                if field == 'Sff_mesh':
                    row['point_fraction_zero_Xi'] = float(np.mean(a == 0))
                    row['volume_fraction_zero_Xi'] = '' if rw is None else float(np.average(a == 0, weights=rw))
                if field in ('hs_length', 'fiber_stretch_raw'):
                    shortening = None if previous is None else a < previous[region]
                    row['point_fraction_shortening'] = '' if shortening is None else float(np.mean(shortening))
                    row['volume_fraction_shortening'] = '' if rw is None or shortening is None else float(np.average(shortening, weights=rw))
                if region_writer is None:
                    region_writer = csv.DictWriter(rh, fieldnames=list(row)); region_writer.writeheader()
                region_writer.writerow(row)
                hot = {'time': t, 'cycle': cycle, 'phase': phase}
                hot.update(('dof_%s' % gid, float(values[i])) for gid, i in zip(args.hotspots, hotspot_indices))
                if hotspot_writer is None:
                    hotspot_writer = csv.DictWriter(hh, fieldnames=list(hot)); hotspot_writer.writeheader()
                hotspot_writer.writerow(hot)
                key = (cycle, phase)
                if key not in accum:
                    accum[key] = {'first': values.copy(), 'last': values.copy(), 'min': values.copy(),
                                  'max': values.copy(), 'regional_sum': 0., 'n': 0,
                                  'regional_min': float(a.min()), 'regional_max': float(a.max()),
                                  'minimum_time': t, 'maximum_time': t}
                state = accum[key]
                state['last'] = values.copy()
                state['min'] = np.minimum(state['min'], values)
                state['max'] = np.maximum(state['max'], values)
                mean = row.get('volume_weighted_mean', row['point_mean'])
                state['regional_sum'] += mean; state['n'] += 1
                if a.min() < state['regional_min']:
                    state['regional_min'] = float(a.min()); state['minimum_time'] = t
                if a.max() > state['regional_max']:
                    state['regional_max'] = float(a.max()); state['maximum_time'] = t
                if field == 'fiber_stretch_raw' and (values.min() < args.low_stretch or values.max() > args.high_stretch):
                    for index in np.flatnonzero((values < args.low_stretch) | (values > args.high_stretch)):
                        if len(anomalies) < args.max_events:
                            anomalies.append({'time': t, 'cycle': cycle, 'phase': phase, 'global_dof': ids[index],
                                              'stretch': float(values[index]), 'in_region': bool(index in region)})
                # Only bounded plot samples; numerical CSVs always retain every aligned row.
                if len(series) < args.plot_points and grid_index % max(1, len(labels)//args.plot_points+1) == 0:
                    series.append((t, mean, float(row['minimum']), float(row['maximum']),
                                   [float(values[i]) for i in hotspot_indices], cycle, phase))
                previous = values
        for (cycle, phase), state in sorted(accum.items()):
            item = {'field': field, 'cycle': cycle, 'phase': phase, 'samples': state['n'],
                    'regional_spatial_mean_time_sample_mean': state['regional_sum']/state['n'],
                    'spatial_mean_kind': 'reference_volume_weighted' if rw is not None else 'unweighted_point_mean',
                    'regional_minimum': state['regional_min'], 'regional_maximum': state['regional_max'],
                    'minimum_time': state['minimum_time'], 'maximum_time': state['maximum_time']}
            # Per-point extrema and signed changes, including every selected point and hotspots.
            point_rows = []
            for index in sorted(set(region.tolist()+hotspot_indices)):
                point_row = {'global_dof': ids[index], 'minimum': float(state['min'][index]),
                                   'maximum': float(state['max'][index]),
                                   'first_sample': float(state['first'][index]), 'last_sample': float(state['last'][index]),
                                   'last_minus_first': float(state['last'][index]-state['first'][index]),
                                   'maximum_drop_from_first_sample': float(state['first'][index]-state['min'][index])}
                point_rows.append(point_row)
                point_comparisons.setdefault((field, phase, ids[index]), []).append((cycle, point_row))
            write_csv(out / ('%s_cycle%d_%s_points.csv' % (field, cycle, phase)), point_rows)
            summaries.append(item)
        plots[field] = series
    changed = [p for p, signature in before.items() if (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != signature]
    if changed:
        raise ValueError('source files changed during analysis; discard output and use an immutable snapshot: ' + str(changed))
    write_csv(out / 'phase_summary.csv', summaries)
    point_variability = []
    for (field, phase, gid), records in sorted(point_comparisons.items()):
        records.sort()
        for metric in ('minimum', 'maximum', 'last_minus_first', 'maximum_drop_from_first_sample'):
            values = np.asarray([r[1][metric] for r in records])
            item = {'field': field, 'phase': phase, 'global_dof': gid, 'metric': metric,
                    'cycles': len(values), 'between_cycle_std': float(values.std()),
                    'between_cycle_range': float(np.ptp(values))}
            for j, difference in enumerate(np.diff(values)):
                item['cycle_%d_to_%d_difference' % (j+1, j+2)] = float(difference)
            point_variability.append(item)
    write_csv(out / 'point_cycle_variability.csv', point_variability)
    variability = []
    for field in paths:
        for phase in PHASES:
            values = [r['regional_spatial_mean_time_sample_mean'] for r in summaries if r['field'] == field and r['phase'] == phase]
            variability.append({'field': field, 'phase': phase, 'cycles': len(values),
                                'mean': float(np.mean(values)), 'between_cycle_std': float(np.std(values)),
                                'between_cycle_range': float(np.ptp(values)),
                                'consecutive_differences': np.diff(values).tolist()})
    with open(out / 'cycle_variability.json', 'w') as handle:
        json.dump(variability, handle, indent=2)
    payload = {'complete': True, 'cycles': selected, 'rejected_cycles': rejected,
                   'region_points': len(region), 'total_points': len(ids), 'hotspots': args.hotspots,
                   'mean_kind': 'reference_volume_weighted' if rw is not None else 'unweighted_point_mean',
                   'percentiles_kind': 'unweighted spatial point percentiles', 'coverage_rows': coverage,
                   'missing_optional_fields': [f for f in OPTIONAL if f not in paths],
                   'geometry_report_sha256': digest(geometry_path), 'source_signatures': before,
                   'mapping_manifest': manifest,
                   'available_global_end_time': float(g['time'][-1]),
                   'last_common_saved_time': float(g['time'][max(common)]),
                   'selected_window': [selected[0]['times'][0], selected[-1]['times'][-1]],
                   'window_policy': ('latest %d complete aligned cycles' if args.start is None else 'first %d complete aligned cycles after explicit start') % args.cycles,
                   'selected_before_calibration_40s': selected[0]['times'][0] < 40.,
                   'runtime': {'python_executable': sys.executable, 'python_version': sys.version, 'numpy_version': np.__version__},
                   'anomaly_thresholds_are_screening_only': [args.low_stretch, args.high_stretch],
                   'anomaly_events': anomalies, 'anomaly_events_may_be_truncated_at': args.max_events,
                   'analysis_parameters': {'axis': args.axis, 'start': args.start,
                       'end': None if not np.isfinite(args.end) else args.end, 'cycles': args.cycles,
                       'flow_threshold': args.flow_threshold, 'volume_rate_threshold': args.volume_rate_threshold,
                       'time_tolerance': args.time_tolerance, 'plot_points': args.plot_points},
                   'limitations': ['Active stress is the saved field, not independently verified equilibrium stress.',
                       'Original FE displacement/F/J and neighbor checks remain necessary for physical conclusions.',
                       'No smoothing, interpolation, timing relabeling or unit conversion is applied.',
                       'Phase intervals are left-closed/right-open; point changes use first/last interior samples.',
                       'Shortening means a negative step in HSL/stretch, separate from stretch below one.',
                       'Sample means are not time-integrated means for irregular sampling.']}
    # Reuse the established standalone SVG renderer; no Matplotlib dependency.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from diagnose_midwall_sarcomere_stdlib import svg_file
    panels = []
    for field, rows in plots.items():
        panel = [('regional mean', [(r[0], r[1]) for r in rows], '#1768ac'),
                 ('regional minimum', [(r[0], r[2]) for r in rows], '#d25435'),
                 ('regional maximum', [(r[0], r[3]) for r in rows], '#16826d')]
        panels.append((field + ' (saved units)', panel))
        hotspot_colors = ('#1768ac', '#d25435', '#16826d', '#8540a0', '#7b5c30')
        svg_file(out / (field + '_hotspots.svg'), [(field + ' (saved units)',
                 [('DOF %d' % gid, [(r[0], r[4][j]) for r in rows], hotspot_colors[j % len(hotspot_colors)])
                  for j, gid in enumerate(args.hotspots)])], field, width=max(1100, 240*len(args.hotspots)+125))
        phase_panels = []
        colors = ('#1768ac', '#d25435', '#16826d', '#8540a0', '#7b5c30')
        for pi, phase in enumerate(PHASES):
            phase_series = []
            for ci, cycle in enumerate(selected):
                start, end = cycle['times'][pi:pi+2]
                xy = [((r[0]-start)/(end-start), r[1]) for r in rows if r[5] == ci+1 and r[6] == phase]
                phase_series.append(('cycle %d' % (ci+1), xy, colors[ci % len(colors)]))
            phase_panels.append((phase.replace('_', ' '), phase_series))
        svg_file(out / (field + '_phase_cycles.svg'), phase_panels,
                 field + ': regional mean aligned to detected events',
                 xlabel='Fraction of detected cardiac phase (plot samples only; no interpolation)', width=1500)
    svg_file(out / 'regional_overview.svg', panels, 'Regional mechanics: ' + ('volume weighted' if rw is not None else 'unweighted point mean'))
    svg_file(out / 'cardiovascular.svg', [(key, [(key, list(zip(g['time'], g[key])), '#1768ac')])
             for key in GLOBAL if key in g and key not in ('time', 'new_beat')], 'Saved cardiovascular signals')
    svg_file(out / 'pressure_volume.svg', [('Ventricular pressure (saved units)',
             [('cycle %d' % (ci+1), list(zip(g['volume_ventricle'][c['indices'][0]:c['indices'][-1]+1],
                                            g['pressure_ventricle'][c['indices'][0]:c['indices'][-1]+1])),
               ('#1768ac', '#d25435', '#16826d', '#8540a0', '#7b5c30')[ci % 5]) for ci, c in enumerate(selected)])],
             'Pressure-volume loops', xlabel='Ventricular volume (saved units)', width=1500)
    # Completion marker is written only after numerical reports and plots succeed.
    if any((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != signature for p, signature in before.items()):
        raise ValueError('sources changed before completion; use an immutable snapshot')
    with open(out / 'report.json', 'w') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    return out


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True, help='sealed directory containing all mapping reports')
    p.add_argument('--input-json', type=Path, help='actual archived submitted input; default DATA/../instruction.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--axis', type=int, choices=(0, 1, 2), required=True, help='verified anatomical long axis')
    p.add_argument('--cycles', type=int, default=5)
    p.add_argument('--start', type=float, help='explicit lower bound; default selects latest complete cycles available')
    p.add_argument('--end', type=float, default=float('inf'))
    p.add_argument('--hotspots', type=int, nargs='+', default=[2833, 2497, 4844, 1580])
    p.add_argument('--time-tolerance', type=float, default=1e-9)
    p.add_argument('--flow-threshold', type=float, default=1e-8)
    p.add_argument('--volume-rate-threshold', type=float, default=1e-5)
    p.add_argument('--low-stretch', type=float, default=.5)
    p.add_argument('--high-stretch', type=float, default=1.35)
    p.add_argument('--max-events', type=int, default=1000)
    p.add_argument('--plot-points', type=int, default=2000)
    return p


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.plot_points < 1 or arguments.max_events < 1 or arguments.time_tolerance <= 0:
        raise SystemExit('plot-points, max-events and time-tolerance must be positive')
    print('Reports:', analyze(arguments))
