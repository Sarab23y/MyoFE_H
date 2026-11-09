#!/usr/bin/env python3
"""MyoFE regional (control-volume) trend viewer. Python 3.8+; numpy and matplotlib only.

WHY THIS SCRIPT EXISTS
-----------------------
check_baseline_trends.py plots a handful of individual mid-wall / mid-ventricle
DOFs (e.g. 1477, 577, 4629 ...). That showed sharp, sometimes contradictory
point-to-point transients. This script answers the follow-up question:
"can I look at a unit of VOLUME instead of a few points, still restricted to
the mid-wall / mid-ventricle region?"

It does that by:
  1. Selecting EVERY material point/DOF that falls inside an explicit
     mid-wall-depth band AND an explicit mid-ventricle (normalized height)
     band -- the same bands you used before to get candidate pools of
     304 / 464 points (0.3-0.7 wall-depth, 0.3-0.7 zfrac by default).
  2. For every requested spatial field, reading ALL of those DOF columns
     from the existing per-field CSVs (no simulation rerun).
  3. Collapsing that whole set of points, at every saved timestep, into a
     single regional trace: a (volume- or point-) weighted MEAN, plus the
     MIN-MAX spread and the standard deviation across the region. This is a
     true spatial average over the selected control volume, not a single
     point's history.
  4. Plotting the regional mean with a shaded spread band (so you see both
     the "typical" regional behavior and how much point-to-point scatter
     remains), and running the same beat-to-beat repeatability check used
     in check_baseline_trends.py, but on the regional mean trace.

Input files are NEVER modified. CSVs must have an explicit time column in
seconds. Spatial CSVs must identify global DOFs in their headers (e.g. 1477
or DOF_1477). No positional DOF guesses, invented times, smoothing, or
interpolation are used beyond the same conservative gap-insertion policy as
check_baseline_trends.py.

HOW TO SELECT THE REGION (pick ONE)
------------------------------------
A) --geometry-csv PATH  (+ optionally --wall-range LO HI --z-range LO HI)
   A CSV with one row per DOF/point giving its wall-depth and normalized
   ventricular-height (zfrac) coordinates -- the same quantities your
   point-selector script (select_midwall_midventricle_*.py) printed/saved
   for all 5008 points before it narrowed down to 8. Column names are
   auto-detected (DOF/global_dof/id, wall/wall_depth, zfrac/normalized_z,
   and an optional volume/weight column). Use --list-geometry-columns to
   see what was detected if auto-detection fails.

B) --dofs-file PATH
   A CSV that already lists the DOFs you want in the region (e.g. a saved
   candidates file with 304/464/840 points). Every DOF in its first integer
   -like column is used directly; no further wall/z filtering is applied.

C) --dofs ID1 ID2 ID3 ...
   A manual list of DOF IDs (mainly for testing / small sanity checks).

EXAMPLES
--------
Using the geometry/candidate file, default 0.3-0.7 / 0.3-0.7 bands:
    python check_region_trends.py --folder sim_output \
        --geometry-csv point_locations.csv --save-plots --no-show

Using an already-saved candidates CSV (no extra filtering):
    python check_region_trends.py --folder sim_output \
        --dofs-file midwall_candidates_304.csv --save-plots --no-show

Output: a new timestamped review folder (region_trend_review_...) with
region_quality_report.json, region_selection.csv (every DOF that was
included, with its wall/zfrac if known), region_beat_statistics.csv,
region_last_beats_summary.csv, and (with --save-plots) PNGs.
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import math
import re
import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Install the plotting packages in your VS Code / container Python environment:\n"
                      "python -m pip install numpy matplotlib")

# ---------------------------------------------------------------------------
# Field catalogue (same spatial / global fields as check_baseline_trends.py)
# ---------------------------------------------------------------------------
SPATIAL = [
    ('Ca_cytosol', 'Calcium in cytosol'), ('Ca_SR', 'Calcium in SR store'),
    ('J_release', 'Calcium release flux'), ('J_uptake', 'Calcium uptake flux'),
    ('Ca_content', 'Total calcium content'),
    ('active_stress', 'Active stress'), ('hs_length', 'Half-sarcomere length'),
    ('fiber_strain', 'Fiber strain'), ('fiber_stretch_raw', 'Fiber stretch'),
    ('Sff_mesh', 'Xi field: Sff_mesh (verify definition)'),
    ('myofiber_passive', 'Myofiber passive stress: myofiber_passive'),
    ('ecm_instantaneous_fiber_stress', 'ECM fiber-direction stress'),
    ('k_act', 'Actual calcium activation rate: k_act'),
    ('k_serca', 'Actual uptake parameter: k_serca'),
    ('k_1', 'Actual myofilament parameter: k_1'),
    ('k_on', 'Actual myofilament parameter: k_on'),
]
GLOBAL = {
    'heart_rate': 'Heart rate', 'baro_b': 'Baroreflex signal: baro_b',
    'baro_c': 'Baroreflex signal: baro_c',
    'arterioles_resistance': 'Arteriolar resistance',
    'veins_compliance': 'Venous compliance',
    't_quiescent_period': 'Quiescent period',
    'pressure_ventricle': 'LV pressure', 'pressure_aorta': 'Aortic pressure',
    'volume_ventricle': 'LV volume', 'baroreflex_active': 'Baroreflex active',
    'new_beat': 'Beat marker', 'activation': 'Activation',
    'ecm_baseline_stable': 'ECM baseline stability flag',
    'ecm_baseline_max_relative_change': 'ECM baseline relative change',
    'ecm_reference_provisional': 'ECM provisional reference flag',
}
DEFAULT_WALL_RANGE = (0.3, 0.7)
DEFAULT_Z_RANGE = (0.3, 0.7)

# ---------------------------------------------------------------------------
# Small helpers (same conservative-parsing philosophy as check_baseline_trends)
# ---------------------------------------------------------------------------

def number(s):
    try:
        value = float(s)
        return value if math.isfinite(value) else float('nan')
    except (ValueError, TypeError):
        return float('nan')


def dof_label(s):
    m = re.fullmatch(r'(?:dof[_ ]?|point[_ ]?|global_dof[_ ]?)?(\d+)', s.strip(), re.I)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Region selection
# ---------------------------------------------------------------------------

def _detect_column(header, candidates):
    low = [h.strip().lower() for h in header]
    for cand in candidates:
        if cand in low:
            return low.index(cand)
    for i, h in enumerate(low):
        if any(cand in h for cand in candidates):
            return i
    return None


def load_geometry(path):
    """Read a per-point geometry/candidate CSV. Returns dict dof -> dict(wall, zfrac, weight)."""
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        header_line = f.readline()
        delimiter = ';' if header_line.count(';') > header_line.count(',') else ','
        header = [h.strip() for h in next(csv.reader([header_line], delimiter=delimiter))]
        dof_i = _detect_column(header, ['dof', 'global_dof', 'point_id', 'id', 'point'])
        wall_i = _detect_column(header, ['wall', 'wall_depth', 'walldepth', 'wall_fraction'])
        z_i = _detect_column(header, ['zfrac', 'z_frac', 'normalized_z', 'height_frac', 'z_normalized'])
        w_i = _detect_column(header, ['volume', 'weight', 'vol', 'quad_weight'])
        if dof_i is None:
            raise ValueError('Could not find a DOF/point-id column in {}. Header was: {}'.format(path.name, header))
        points = {}
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row or len(row) <= dof_i:
                continue
            dof = dof_label(row[dof_i].strip())
            if dof is None:
                dof = int(number(row[dof_i])) if math.isfinite(number(row[dof_i])) else None
            if dof is None:
                continue
            points[dof] = dict(
                wall=number(row[wall_i]) if wall_i is not None and wall_i < len(row) else float('nan'),
                zfrac=number(row[z_i]) if z_i is not None and z_i < len(row) else float('nan'),
                weight=number(row[w_i]) if w_i is not None and w_i < len(row) else float('nan'),
            )
        if not points:
            raise ValueError('No usable rows found in geometry file {}'.format(path.name))
        return points, dict(dof_col=header[dof_i], wall_col=header[wall_i] if wall_i is not None else None,
                             zfrac_col=header[z_i] if z_i is not None else None,
                             weight_col=header[w_i] if w_i is not None else None)


def load_dofs_file(path):
    """Read a CSV that already lists candidate DOFs (any column with integer-like IDs)."""
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        header_line = f.readline()
        delimiter = ';' if header_line.count(';') > header_line.count(',') else ','
        header = [h.strip() for h in next(csv.reader([header_line], delimiter=delimiter))]
        dof_i = _detect_column(header, ['dof', 'global_dof', 'point_id', 'id', 'point'])
        if dof_i is None:
            dof_i = 0
        dofs = []
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row or len(row) <= dof_i:
                continue
            dof = dof_label(row[dof_i].strip())
            if dof is None and math.isfinite(number(row[dof_i])):
                dof = int(number(row[dof_i]))
            if dof is not None:
                dofs.append(dof)
        if not dofs:
            raise ValueError('No DOF IDs found in {}'.format(path.name))
        return sorted(set(dofs))


def select_region_dofs(points, wall_range, z_range):
    lo_w, hi_w = wall_range
    lo_z, hi_z = z_range
    selected, skipped_missing = [], 0
    for dof, info in points.items():
        w, z = info['wall'], info['zfrac']
        if not (math.isfinite(w) and math.isfinite(z)):
            skipped_missing += 1
            continue
        if lo_w <= w <= hi_w and lo_z <= z <= hi_z:
            selected.append(dof)
    return sorted(selected), skipped_missing


# ---------------------------------------------------------------------------
# CSV loading: same robust streaming policy as check_baseline_trends.py,
# but selecting an arbitrary (potentially large) set of DOF columns.
# ---------------------------------------------------------------------------

def load_csv(path, dofs, max_rows=None):
    report = dict(file=str(path), warnings=[], examples=[], rows_seen=0,
                  malformed_rows=0, invalid_times=0, invalid_cells=0,
                  missing_dofs=[], stopped_at_time_reset=False)
    times, values = [], []
    last = None
    pending_gap = False
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
        header_line = f.readline()
        delimiter = ';' if header_line.count(';') > header_line.count(',') else ','
        header = [s.strip() for s in next(csv.reader([header_line], delimiter=delimiter))]
        time_cols = [i for i, h in enumerate(header) if h.lower() in ('time', 'time_s', 'time (s)')]
        if len(time_cols) != 1:
            raise ValueError('Need one explicit time column; found {}'.format(len(time_cols)))
        ti = time_cols[0]
        lookup = {}
        for i, h in enumerate(header):
            gid = dof_label(h)
            if i != ti and gid is not None:
                if gid in lookup:
                    raise ValueError('Duplicate DOF label: {}'.format(gid))
                lookup[gid] = i
        report['missing_dofs'] = [d for d in dofs if d not in lookup]
        names = [str(d) for d in dofs if d in lookup]
        cols = [lookup[int(d)] for d in names]
        if not names:
            raise ValueError('None of the {} requested region DOFs were found in this file\'s header'.format(len(dofs)))
        for lineno, line in enumerate(f, 2):
            if not line.strip():
                continue
            if max_rows is not None and report['rows_seen'] >= max_rows:
                report['warnings'].append('Stopped at metadata saved-row count {}; later downloaded rows conservatively excluded.'.format(max_rows))
                break
            report['rows_seen'] += 1
            if not line.endswith(('\n', '\r')):
                report['malformed_rows'] += 1
                pending_gap = True
                report['warnings'].append('Final unterminated line omitted conservatively (possible interrupted copy).')
                break
            try:
                row = next(csv.reader([line], delimiter=delimiter, strict=True))
            except csv.Error:
                row = []
            if len(row) != len(header):
                report['malformed_rows'] += 1
                pending_gap = True
                if len(report['examples']) < 10:
                    report['examples'].append('Malformed row at line {}'.format(lineno))
                continue
            t = number(row[ti])
            if not math.isfinite(t) or t < 0:
                report['invalid_times'] += 1
                pending_gap = True
                continue
            if last is not None and t <= last:
                report['stopped_at_time_reset'] = True
                report['warnings'].append('Stopped at duplicate/decreasing time on line {}; '
                                           'later rows excluded (possible unwritten tail or restart).'.format(lineno))
                break
            if pending_gap and last is not None:
                times.append((last + t) / 2)
                values.append([float('nan')] * len(names))
            v = [number(row[i]) for i in cols]
            report['invalid_cells'] += sum(not math.isfinite(x) for x in v)
            times.append(t)
            values.append(v)
            last = t
            pending_gap = False
            if report['rows_seen'] % 20000 == 0:
                print('  {}: {:,} rows scanned'.format(path.name, report['rows_seen']), flush=True)
    if len(times) < 2:
        raise ValueError('Fewer than two usable time rows')
    t, y = np.asarray(times), np.asarray(values, dtype=float)
    report.update(first_time=float(t[0]), last_time=float(t[-1]), retained_rows=len(t), columns=names,
                  trailing_malformed_or_invalid_rows=bool(pending_gap))
    return {'t': t, 'y': y, 'names': names, 'report': report}


def get_trace(table, name):
    if table is None or name not in table['names']:
        return None
    return table['y'][:, table['names'].index(name)]


def choose_folder(arg):
    if arg:
        return Path(arg).expanduser()
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title='Select downloaded sim_output folder')
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass
    return Path(input('Paste the sim_output folder path: ').strip().strip('"'))


def plot_ready(t, y):
    if len(t) < 3:
        return t, y
    dt = float(np.median(np.diff(t)))
    jumps = np.flatnonzero(np.diff(t) > 1.6 * dt) + 1
    if len(jumps):
        tnew = np.insert(t, jumps, (t[jumps - 1] + t[jumps]) / 2)
        ynew = np.insert(y, jumps, np.nan, axis=0)
        return tnew, ynew
    return t, y


# ---------------------------------------------------------------------------
# Regional aggregation: collapse N DOF columns -> mean / min / max / std per row
# ---------------------------------------------------------------------------

def aggregate_region(table, weights=None):
    """Row-wise statistics across the region's columns. Returns dict with
    mean, min, max, std, n_valid arrays aligned to table['t']."""
    y = table['y']
    if weights is not None:
        w = np.array([weights.get(int(name), np.nan) for name in table['names']])
        if np.any(np.isfinite(w)) and np.nansum(np.where(np.isfinite(w), w, 0)) > 0:
            w = np.where(np.isfinite(w), w, 0.0)
            masked = np.where(np.isfinite(y), y, 0.0)
            valid = np.isfinite(y).astype(float)
            denom = (valid * w[None, :]).sum(axis=1)
            mean = np.divide((masked * w[None, :]).sum(axis=1), denom, out=np.full(y.shape[0], np.nan), where=denom > 0)
        else:
            mean = np.nanmean(y, axis=1)
    else:
        mean = np.nanmean(y, axis=1)
    with np.errstate(invalid='ignore'):
        lo = np.nanmin(y, axis=1)
        hi = np.nanmax(y, axis=1)
        std = np.nanstd(y, axis=1)
    n_valid = np.isfinite(y).sum(axis=1)
    return {'t': table['t'], 'mean': mean, 'min': lo, 'max': hi, 'std': std, 'n_valid': n_valid}


def draw_region(ax, agg, lo, hi):
    t, mean, ymin, ymax = agg['t'], agg['mean'], agg['min'], agg['max']
    mask = (t >= lo) & (t <= hi)
    tt, mm = plot_ready(t[mask], mean[mask][:, None])
    _, lo_arr = plot_ready(t[mask], ymin[mask][:, None])
    _, hi_arr = plot_ready(t[mask], ymax[mask][:, None])
    ax.fill_between(tt, lo_arr[:, 0], hi_arr[:, 0], color='tab:blue', alpha=0.18,
                     label='Region min-max', zorder=1)
    ax.plot(tt, mm[:, 0], lw=1.3, color='tab:blue', label='Region mean', zorder=3)
    ax.set_xlim(lo, hi if hi > lo else lo + 1)
    ax.grid(alpha=0.22)
    ax.set_xlabel('Simulation time (s)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
    for event, color in ((5, '#a76c00'), (40, '#855ca5')):
        if lo <= event <= hi:
            ax.axvline(event, ls=':', color=color, lw=1)


def draw_global(ax, table, lo, hi):
    t, y = table['t'], table['y']
    mask = (t >= lo) & (t <= hi)
    tt, yy = plot_ready(t[mask], y[mask])
    ax.plot(tt, yy[:, 0], lw=1.1, color='tab:blue', zorder=3)
    ax.set_xlim(lo, hi if hi > lo else lo + 1)
    ax.grid(alpha=0.22)
    ax.set_xlabel('Simulation time (s)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
    for event, color in ((5, '#a76c00'), (40, '#855ca5')):
        if lo <= event <= hi:
            ax.axvline(event, ls=':', color=color, lw=1)


def panels_region(title, items, args, figures):
    for start in range(0, len(items), 4):
        chunk = items[start:start + 4]
        if not chunk:
            continue
        fig, axes = plt.subplots(len(chunk), 2, figsize=(13, 2.5 * len(chunk) + 1), squeeze=False)
        last = max(float(agg['t'][-1]) for _, agg, _ in chunk)
        first = min(float(agg['t'][0]) for _, agg, _ in chunk)
        for row, (label, agg, n_points) in enumerate(chunk):
            end = float(agg['t'][-1])
            for col, (lo, hi) in enumerate(((first, last), (max(first, last - args.last_seconds), last))):
                draw_region(axes[row, col], agg, lo, hi)
                axes[row, col].set_ylabel(label + '\n(file units, n={})'.format(n_points), fontsize=9)
                if end < hi - 0.002:
                    axes[row, col].axvspan(max(lo, end), hi, color='0.92')
                    axes[row, col].text(.98, .92, 'Data ends at {:.3f} s'.format(end), transform=axes[row, col].transAxes,
                                        ha='right', va='top', fontsize=8)
            handles, labels = axes[row, 1].get_legend_handles_labels()
            if handles:
                axes[row, 1].legend(handles, labels, fontsize=7, loc='upper right')
        axes[0, 0].set_title('Entire available recording')
        axes[0, 1].set_title('Most recent {:g} seconds'.format(args.last_seconds))
        fig.suptitle(title + ' | regional (control-volume) average, preliminary data', fontsize=13)
        fig.text(.5, .01, 'Blue line: mean across all selected region points. Shaded band: point-to-point '
                 'min-max spread at that instant. Gaps = unavailable/invalid data. Dotted times: 5 s baroreflex, '
                 '40 s ECM reference.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, .95))
        figures.append((title.replace(' ', '_') + '_' + str(start // 4 + 1), fig))


def panels_global(title, items, args, figures):
    for start in range(0, len(items), 4):
        chunk = items[start:start + 4]
        if not chunk:
            continue
        fig, axes = plt.subplots(len(chunk), 2, figsize=(13, 2.5 * len(chunk) + 1), squeeze=False)
        last = max(float(tab['t'][-1]) for _, tab in chunk)
        first = min(float(tab['t'][0]) for _, tab in chunk)
        for row, (label, table) in enumerate(chunk):
            end = float(table['t'][-1])
            for col, (lo, hi) in enumerate(((first, last), (max(first, last - args.last_seconds), last))):
                draw_global(axes[row, col], table, lo, hi)
                axes[row, col].set_ylabel(label + '\n(file units)', fontsize=9)
                if end < hi - 0.002:
                    axes[row, col].axvspan(max(lo, end), hi, color='0.92')
        axes[0, 0].set_title('Entire available recording')
        axes[0, 1].set_title('Most recent {:g} seconds'.format(args.last_seconds))
        fig.suptitle(title + ' | whole-model signal (not region-specific)', fontsize=13)
        fig.tight_layout(rect=(0, .02, 1, .95))
        figures.append((title.replace(' ', '_') + '_' + str(start // 4 + 1), fig))


def beat_intervals(global_table):
    marker = get_trace(global_table, 'new_beat')
    if marker is None:
        return []
    t = global_table['t']
    on = marker > .5
    valid = np.isfinite(marker)
    rising = np.flatnonzero(on[1:] & ~on[:-1] & valid[1:] & valid[:-1]) + 1
    edges = t[rising]
    dt = np.median(np.diff(t))
    result = []
    for a, b in zip(edges[:-1], edges[1:]):
        inside = (t >= a) & (t <= b)
        if np.all(valid[inside]) and np.max(np.diff(t[inside])) <= 1.6 * dt:
            result.append((float(a), float(b)))
    return result


def cycle_values(t, y, a, b):
    dt = float(np.median(np.diff(t)))
    if a < t[0] - dt * .1 or b > t[-1] + dt * .1:
        return None
    mask = (t >= a - dt * .1) & (t <= b + dt * .1)
    tt, yy = t[mask], y[mask]
    if len(tt) < 3 or not np.isfinite(yy).all():
        return None
    if tt[0] - a > dt * 1.1 or b - tt[-1] > dt * 1.1 or np.max(np.diff(tt)) > 1.6 * dt:
        return None
    return tt, yy


def write_region_beats(region_series, intervals, out, args, figures):
    records, summaries = [], []
    for field, agg in region_series.items():
        t, y = agg['t'], agg['mean']
        rows = []
        for bi, (a, b) in enumerate(intervals):
            vals = cycle_values(t, y, a, b)
            if vals is None:
                continue
            tt, vv = vals
            mean = float(np.sum((vv[1:] + vv[:-1]) * .5 * np.diff(tt)) / (tt[-1] - tt[0]))
            item = dict(field=field, beat_index=bi, start_s=a, end_s=b,
                        minimum=float(vv.min()), maximum=float(vv.max()), mean=mean)
            records.append(item)
            rows.append(item)
        eligible = [i for i, (_, b) in enumerate(intervals) if b <= t[-1] + 1e-9]
        target = eligible[-args.last_beats:]
        tail = [r for r in rows if r['beat_index'] in target]
        if len(target) < args.last_beats or len(tail) != args.last_beats:
            continue
        scale = max(max(abs(r['minimum']), abs(r['maximum'])) for r in tail)
        s = dict(field=field, beats=len(tail), start_s=tail[0]['start_s'], end_s=tail[-1]['end_s'])
        for metric in ('minimum', 'maximum', 'mean'):
            v = [r[metric] for r in tail]
            s[metric + '_spread_percent_of_signal_scale'] = 100 * (max(v) - min(v)) / scale if scale > 1e-30 else ''
        s['normalization_scale'] = scale
        summaries.append(s)
    for filename, rows in [('region_beat_statistics.csv', records), ('region_last_beats_summary.csv', summaries)]:
        if rows:
            with (out / filename).open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    plotrows = []
    for field in region_series:
        rr = [r for r in records if r['field'] == field]
        if rr:
            plotrows.append((field, rr))
    for start in range(0, len(plotrows), 4):
        chunk = plotrows[start:start + 4]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(11, 2.5 * len(chunk) + 1), squeeze=False)
        for ax, (field, rr) in zip(axes[:, 0], chunk):
            xx, low, high, avg = [], [], [], []
            prev = None
            for r in rr:
                if prev is not None and r['beat_index'] != prev + 1:
                    xx.append(r['end_s']); low.append(np.nan); high.append(np.nan); avg.append(np.nan)
                xx.append(r['end_s']); low.append(r['minimum']); high.append(r['maximum']); avg.append(r['mean'])
                prev = r['beat_index']
            ax.plot(xx, high, '-o', ms=3, label='Beat maximum', color='#c34b32')
            ax.plot(xx, low, '-o', ms=3, label='Beat minimum', color='#286ca8')
            ax.plot(xx, avg, '-', label='Beat mean', color='#397d39')
            ax.set_ylabel(field + '\n(regional mean, file units)', fontsize=9)
            ax.set_xlabel('Beat end time (s)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('Are successive beats becoming repeatable? | REGIONAL mean (mid-wall/mid-ventricle control volume)')
        fig.tight_layout(rect=(0, 0, 1, .95))
        figures.append(('Region_beat_trends_' + str(start // 4 + 1), fig))
    return len(records), len(summaries)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--geometry-csv', help='Per-DOF wall-depth + zfrac (+ optional weight) table for ALL candidate points')
    src.add_argument('--dofs-file', help='CSV already listing the DOFs to include in the region')
    src.add_argument('--dofs', nargs='+', type=int, help='Manual list of DOF IDs to include in the region')
    ap.add_argument('--wall-range', nargs=2, type=float, default=list(DEFAULT_WALL_RANGE),
                     help='Wall-depth band for region selection (used only with --geometry-csv). Default 0.3 0.7')
    ap.add_argument('--z-range', nargs=2, type=float, default=list(DEFAULT_Z_RANGE),
                     help='Normalized ventricular-height (zfrac) band (used only with --geometry-csv). Default 0.3 0.7')
    ap.add_argument('--last-seconds', type=float, default=5)
    ap.add_argument('--last-beats', type=int, default=5)
    ap.add_argument('--end-time', type=float, help='Optional user-chosen cutoff in seconds; keeps earlier values only')
    ap.add_argument('--save-plots', action='store_true')
    ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--list-geometry-columns', action='store_true',
                     help='Print the auto-detected geometry column names and exit (debugging aid)')
    args = ap.parse_args()
    if args.last_seconds <= 0 or args.last_beats < 2:
        ap.error('last-seconds must be positive and last-beats at least 2')

    weights = None
    region_dofs, selection_report = [], {}
    if args.geometry_csv:
        points, detected = load_geometry(Path(args.geometry_csv).expanduser())
        if args.list_geometry_columns:
            print('Detected columns:', detected)
            return
        region_dofs, n_missing = select_region_dofs(points, tuple(args.wall_range), tuple(args.z_range))
        if not region_dofs:
            raise SystemExit('No points fell inside wall-range {} / z-range {}. Check --list-geometry-columns '
                              'output and widen the bands if needed.'.format(args.wall_range, args.z_range))
        if any(math.isfinite(points[d]['weight']) for d in region_dofs):
            weights = {d: points[d]['weight'] for d in region_dofs}
        selection_report = dict(mode='geometry-csv', columns=detected,
                                 wall_range=args.wall_range, z_range=args.z_range,
                                 points_with_missing_wall_or_z=n_missing,
                                 n_region_points=len(region_dofs))
        region_rows = [dict(dof=d, wall=points[d]['wall'], zfrac=points[d]['zfrac']) for d in region_dofs]
    elif args.dofs_file:
        region_dofs = load_dofs_file(Path(args.dofs_file).expanduser())
        selection_report = dict(mode='dofs-file', source=str(args.dofs_file), n_region_points=len(region_dofs))
        region_rows = [dict(dof=d) for d in region_dofs]
    elif args.dofs:
        region_dofs = sorted(set(args.dofs))
        selection_report = dict(mode='manual-list', n_region_points=len(region_dofs))
        region_rows = [dict(dof=d) for d in region_dofs]
    else:
        ap.error('Provide exactly one region source: --geometry-csv, --dofs-file, or --dofs')

    folder = choose_folder(args.folder)
    if (folder / 'sim_output').is_dir() and not (folder / 'data.csv').is_file():
        folder = folder / 'sim_output'
    if not folder.is_dir():
        raise SystemExit('Folder does not exist: {}'.format(folder))
    out = folder / ('region_trend_review_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    with (out / 'region_selection.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(region_rows[0]))
        writer.writeheader()
        writer.writerows(region_rows)

    report = {'folder': str(folder), 'region_selection': selection_report, 'files': {},
              'missing_fields': [], 'errors': [],
              'policy': 'Finite zeros and spikes preserved. Bad cells are NaN gaps. Malformed rows excluded. '
                        'Stop at first duplicate/decreasing timestamp. No interpolation or inferred time/DOF mapping.',
              'aggregation': 'Volume-weighted mean if a weight column was detected in --geometry-csv, otherwise a '
                              'simple arithmetic mean across every included DOF at each saved timestep. Min/max/std '
                              'across the region are also reported so point-to-point scatter is visible, not hidden.',
              'units': 'Values remain in original file units; no stress-measure or concentration conversion.'}
    metadata = None
    try:
        metadata = json.loads((folder / 'output_metadata.json').read_text(encoding='utf-8-sig'))
        report['metadata_complete'] = metadata.get('complete')
        print('Metadata complete flag:', metadata.get('complete'))
    except (OSError, ValueError) as exc:
        report['metadata_note'] = 'Metadata unavailable/unreadable: ' + str(exc)

    inventory = {p.name.lower(): p for p in folder.glob('*.csv')}
    region_series, global_table = {}, None

    def read_one(path, dofs):
        print('Reading {} ({:.1f} MB) ...'.format(path.name, path.stat().st_size / 1024 ** 2), flush=True)
        try:
            key = 'global_rows_written' if dofs is None else 'spatial_rows_written'
            cap = metadata.get(key) if metadata else None
            cap = cap if isinstance(cap, int) and cap >= 0 else None
            tab = load_csv(path, dofs, max_rows=cap) if dofs is not None else load_csv_global(path, max_rows=cap)
            if args.end_time is not None:
                keep = tab['t'] <= args.end_time
                tab['t'], tab['y'] = tab['t'][keep], tab['y'][keep]
            if len(tab['t']) < 2:
                raise ValueError('Not enough samples after requested cutoff')
            report['files'][path.name] = tab['report']
            print('  Plotting {:.3f} to {:.3f} s; {} region columns found'.format(tab['t'][0], tab['t'][-1], len(tab['names'])))
            return tab
        except (OSError, ValueError, csv.Error) as exc:
            report['errors'].append('{}: {}'.format(path.name, exc))
            print('  SKIPPED:', exc)
            return None

    if 'data.csv' in inventory:
        global_table = read_one(inventory['data.csv'], None)
    else:
        report['missing_fields'].append('data.csv (global outputs and heartbeat markers)')

    for field, title in SPATIAL:
        path = inventory.get((field + '_data.csv').lower()) or inventory.get((field + '.csv').lower())
        if path:
            tab = read_one(path, region_dofs)
            if tab is not None:
                region_series[field] = aggregate_region(tab, weights)
                report['files'][path.name]['n_region_points_found'] = len(tab['names'])
                report['files'][path.name]['n_region_points_requested'] = len(region_dofs)
        else:
            report['missing_fields'].append(field)

    global_series = {}
    if global_table is not None:
        for name in GLOBAL:
            v = get_trace(global_table, name)
            if v is not None:
                global_series[name] = {'t': global_table['t'], 'y': v[:, None], 'names': [name]}

    if not region_series and not global_series:
        (out / 'region_quality_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise SystemExit('No usable requested signals. Check folder/CSV headers/region DOFs. Report: {}'.format(out))

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figures = []
    titles = dict(SPATIAL, **GLOBAL)
    region_groups = [
        ('Calcium handling (regional average)', ['Ca_cytosol', 'Ca_SR', 'J_release', 'J_uptake', 'Ca_content']),
        ('Contraction and relaxation (regional average)', ['active_stress', 'hs_length', 'fiber_strain', 'fiber_stretch_raw']),
        ('Passive myofiber and ECM (regional average)', ['Sff_mesh', 'myofiber_passive', 'ecm_instantaneous_fiber_stress']),
        ('Actual kinetic parameters (regional average)', ['k_act', 'k_serca', 'k_1', 'k_on']),
    ]
    for title, fields in region_groups:
        items = [(titles[k], region_series[k], len(report['files'].get(k + '_data.csv', {}).get('columns', region_dofs)))
                 for k in fields if k in region_series]
        panels_region(title, items, args, figures)

    global_groups = [
        ('Baroreflex and circulation', ['heart_rate', 'baro_b', 'baro_c', 'arterioles_resistance', 'veins_compliance', 't_quiescent_period']),
        ('Pressure and volume', ['pressure_ventricle', 'pressure_aorta', 'volume_ventricle']),
        ('ECM reference diagnostics', ['ecm_baseline_stable', 'ecm_baseline_max_relative_change', 'ecm_reference_provisional']),
    ]
    for title, fields in global_groups:
        panels_global(title, [(titles[k], global_series[k]) for k in fields if k in global_series], args, figures)

    intervals = beat_intervals(global_table) if global_table is not None else []
    counts = write_region_beats(region_series, intervals, out, args, figures) if intervals else (0, 0)
    report['complete_beat_intervals_from_new_beat'] = intervals
    report['region_beat_statistics_rows'], report['region_last_beats_summary_rows'] = counts
    report['requested_end_time'] = args.end_time

    notes = [
        'PRELIMINARY REGIONAL (CONTROL-VOLUME) REVIEW OF DOWNLOADED MYOFE OUTPUTS',
        'Input files were not modified. Each file retains its own actual time coverage.',
        'Region = {} points satisfying the selection in region_selection.csv.'.format(len(region_dofs)),
        selection_report.get('mode', '') == 'geometry-csv' and
            'Bands used: wall-depth {} , zfrac {}.'.format(args.wall_range, args.z_range) or
            'DOFs were supplied directly (no wall/z band re-filtered here).',
        'Regional mean is volume-weighted if a weight column was found, otherwise a simple arithmetic mean.',
        'Shaded band on each plot = min-max spread across the region at that instant (point-to-point scatter is '
        'shown, not hidden by averaging).',
        'Metadata saved-row counters, when present, cap reading conservatively; stale metadata can omit later valid downloaded rows.',
        'Zeros are kept: zero active/passive stress can be legitimate.',
        'NaN/Inf/non-numeric values are gaps. Malformed rows are excluded. Missing timestamps create gaps.',
        'First duplicate/decreasing timestamp ends that file: check report for possible padding or restarted recordings.',
        'All values use file units. Verify units and definitions before interpreting magnitude.',
        'Beat plots use explicit new_beat edges on the regional mean trace. First and last unbounded partial cycles are omitted.',
        'last_beats_summary requires the latest {} consecutive complete intervals.'.format(args.last_beats),
        'Spread = 100*(max statistic - min statistic)/largest absolute signal value across those beats (same '
        'normalization policy as check_baseline_trends.py), applied here to the regional mean, not a single point.',
        'No automatic pass/fail plateau or physiological-validity claim is made.',
        'Beat summaries unavailable: need explicit new_beat markers and enough valid complete beats.' if not counts[0] else 'Beat summaries created.',
        'Missing fields: ' + ', '.join(report['missing_fields']),
        'Errors: ' + '; '.join(report['errors']),
    ]
    (out / 'region_quality_report.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding='utf-8')
    (out / 'region_review_notes.txt').write_text('\n'.join(str(n) for n in notes), encoding='utf-8')
    if args.save_plots:
        for name, fig in figures:
            fig.savefig(out / (name + '.png'), dpi=150, bbox_inches='tight')
    print('\nREGION REVIEW FOLDER:', out)
    print('Region points requested: {} | Figures:'.format(len(region_dofs)), len(figures), '| beat-statistic rows:', counts[0])
    print('Missing fields:', ', '.join(report['missing_fields']) or 'none')
    print('Read region_quality_report.json for bad rows, missing DOFs, and actual region coverage per field.')
    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


def load_csv_global(path, max_rows=None):
    """Load data.csv (global + RC columns), same logic as check_baseline_trends.py's dofs=None branch,
    trimmed to only what this script needs (GLOBAL fields)."""
    report = dict(file=str(path), warnings=[], examples=[], rows_seen=0,
                  malformed_rows=0, invalid_times=0, invalid_cells=0, stopped_at_time_reset=False)
    times, values = [], []
    last = None
    pending_gap = False
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
        header_line = f.readline()
        delimiter = ';' if header_line.count(';') > header_line.count(',') else ','
        header = [s.strip() for s in next(csv.reader([header_line], delimiter=delimiter))]
        time_cols = [i for i, h in enumerate(header) if h.lower() in ('time', 'time_s', 'time (s)')]
        if len(time_cols) != 1:
            raise ValueError('Need one explicit time column; found {}'.format(len(time_cols)))
        ti = time_cols[0]
        cols = [i for i, h in enumerate(header) if h in GLOBAL]
        names = [header[i] for i in cols]
        if not names:
            raise ValueError('No requested GLOBAL columns found in data.csv')
        for lineno, line in enumerate(f, 2):
            if not line.strip():
                continue
            if max_rows is not None and report['rows_seen'] >= max_rows:
                report['warnings'].append('Stopped at metadata saved-row count {}.'.format(max_rows))
                break
            report['rows_seen'] += 1
            if not line.endswith(('\n', '\r')):
                report['malformed_rows'] += 1
                pending_gap = True
                break
            try:
                row = next(csv.reader([line], delimiter=delimiter, strict=True))
            except csv.Error:
                row = []
            if len(row) != len(header):
                report['malformed_rows'] += 1
                pending_gap = True
                continue
            t = number(row[ti])
            if not math.isfinite(t) or t < 0:
                report['invalid_times'] += 1
                pending_gap = True
                continue
            if last is not None and t <= last:
                report['stopped_at_time_reset'] = True
                break
            if pending_gap and last is not None:
                times.append((last + t) / 2)
                values.append([float('nan')] * len(names))
            v = [number(row[i]) for i in cols]
            report['invalid_cells'] += sum(not math.isfinite(x) for x in v)
            times.append(t)
            values.append(v)
            last = t
            pending_gap = False
    if len(times) < 2:
        raise ValueError('Fewer than two usable time rows')
    t, y = np.asarray(times), np.asarray(values, dtype=float)
    report.update(first_time=float(t[0]), last_time=float(t[-1]), retained_rows=len(t), columns=names)
    return {'t': t, 'y': y, 'names': names, 'report': report}


if __name__ == '__main__':
    main()
