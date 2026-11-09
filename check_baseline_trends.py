#!/usr/bin/env python3
"""MyoFE partial-run viewer. Python 3.8+; numpy and matplotlib only.

Run in VS Code, select your downloaded sim_output folder, and view the figures.
Or: python check_baseline_trends.py --folder "/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/sim_output"

Input files are NEVER modified. CSVs must have an explicit time column in seconds.
Spatial CSVs must identify global DOFs in their headers (e.g. 1477 or DOF_1477).
No positional DOF guesses, invented times, smoothing, or interpolation are used.
Default: highlight DOF 1477; other selected points are gray context lines.
Use --all-points for colored lines for every selected point.
Output: a new timestamped review folder with quality_report.json,
review_notes.txt, beat_statistics.csv and last_beats_summary.csv when possible.
Use --save-plots to also save PNGs; --no-show for noninteractive execution.
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import math
import re
import sys

# OPTIONAL: set this to your downloaded folder, then click Run in VS Code.
RESULTS_FOLDER = r"/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/sim_output"
DEFAULT_DOFS = [1477, 577, 4629, 4633, 2589, 3041, 1941, 1473]

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Install the plotting packages in your VS Code Python environment:\n"
                     "python -m pip install numpy matplotlib")

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
RC_PREFIXES = ['membranes_k_act_rc_', 'membranes_k_serca_rc_',
               'myofilaments_k_1_rc_', 'myofilaments_k_on_rc_']


def number(s):
    try:
        value = float(s)
        return value if math.isfinite(value) else float('nan')
    except (ValueError, TypeError):
        return float('nan')


def dof_label(s):
    m = re.fullmatch(r'(?:dof[_ ]?|point[_ ]?|global_dof[_ ]?)?(\d+)', s.strip(), re.I)
    return int(m.group(1)) if m else None


def load_csv(path, dofs=None, max_rows=None):
    """Stream rows, retaining requested columns only; malformed rows become gaps."""
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
        rc_groups = {}
        if dofs is None:
            cols = [i for i, h in enumerate(header) if h in GLOBAL]
            names = [header[i] for i in cols]
            rc_groups = {p: [i for i, h in enumerate(header) if h.startswith(p)] for p in RC_PREFIXES}
            rc_groups = {p: ii for p, ii in rc_groups.items() if ii}
            names += [p + 'recorded_mean' for p in rc_groups]
            report['rc_column_counts'] = {p: len(ii) for p, ii in rc_groups.items()}
        else:
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
            raise ValueError('No requested columns found; no column-position assumptions made')
        if len(names) != len(set(names)):
            raise ValueError('Duplicate selected column names')
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
                # Conservatively preserve the first monotonic segment only.
                report['stopped_at_time_reset'] = True
                report['warnings'].append('Stopped at duplicate/decreasing time on line {}; '
                                           'later rows excluded (possible unwritten tail or restart).'.format(lineno))
                break
            if pending_gap and last is not None:
                times.append((last + t) / 2)
                values.append([float('nan')] * len(names))
            v = [number(row[i]) for i in cols]
            for prefix, ii in rc_groups.items():
                rv = [number(row[i]) for i in ii]
                # Streaming aggregation prevents thousands of RC histories filling RAM.
                v.append(sum(x / len(rv) for x in rv) if all(math.isfinite(x) for x in rv) else float('nan'))
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
    report['columns_quality'] = {}
    for j, name in enumerate(names):
        v = y[:, j]
        finite = v[np.isfinite(v)]
        report['columns_quality'][name] = {
            'finite_samples': int(len(finite)), 'zeros': int(np.sum(finite == 0)),
            'negative_samples': int(np.sum(finite < 0)),
            'all_finite_values_zero': bool(len(finite) and np.all(finite == 0)),
            'min': float(finite.min()) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None,
        }
    # Preserve zeros and all finite spikes. Flag suspicious physical values without deleting them.
    stem = path.stem.removesuffix('_data') if sys.version_info >= (3, 9) else path.stem.replace('_data', '')
    if stem in ('Ca_cytosol', 'Ca_SR', 'Ca_content', 'hs_length', 'fiber_stretch_raw'):
        bad = y <= 0 if stem in ('hs_length', 'fiber_stretch_raw') else y < 0
        if np.any(bad):
            report['warnings'].append('{} negative/nonpositive physical values retained for inspection.'.format(int(bad.sum())))
    return {'t': t, 'y': y, 'names': names, 'report': report}


def get_trace(table, name):
    if table is None or name not in table['names']:
        return None
    return table['y'][:, table['names'].index(name)]


def choose_folder(arg):
    if arg or RESULTS_FOLDER:
        return Path(arg or RESULTS_FOLDER).expanduser()
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
    """Insert gaps across missing timestamps; keep the original finite observations."""
    if len(t) < 3:
        return t, y
    dt = float(np.median(np.diff(t)))
    jumps = np.flatnonzero(np.diff(t) > 1.6 * dt) + 1
    if len(jumps):
        tnew = np.insert(t, jumps, (t[jumps - 1] + t[jumps]) / 2)
        ynew = np.insert(y, jumps, np.nan, axis=0)
        return tnew, ynew
    return t, y


def draw_table(ax, table, lo, hi, args):
    t, y = table['t'], table['y']
    mask = (t >= lo) & (t <= hi)
    t, y = plot_ready(t[mask], y[mask])
    for j, name in enumerate(table['names']):
        if len(table['names']) == 1 or args.all_points or name == str(args.focus):
            color = 'tab:blue' if not args.all_points else 'C{}'.format(args.dofs.index(int(name)) % 10) if name.isdigit() else 'C{}'.format(j % 10)
            label = 'DOF ' + name if name.isdigit() else name
            ax.plot(t, y[:, j], lw=1.1, color=color, label=label, zorder=3)
        else:
            ax.plot(t, y[:, j], lw=0.6, color='0.65', alpha=0.7, zorder=1)
    ax.set_xlim(lo, hi if hi > lo else lo + 1)
    ax.grid(alpha=0.22)
    ax.set_xlabel('Simulation time (s)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
    for event, color in ((5, '#a76c00'), (40, '#855ca5')):
        if lo <= event <= hi:
            ax.axvline(event, ls=':', color=color, lw=1)


def panels(title, items, args, figures):
    # At most four rows per window for legibility.
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
                draw_table(axes[row, col], table, lo, hi, args)
                axes[row, col].set_ylabel(label + '\n(file units)', fontsize=9)
                if end < hi - 0.002:
                    axes[row, col].axvspan(max(lo, end), hi, color='0.92')
                    axes[row, col].text(.98, .92, 'Data ends at {:.3f} s'.format(end), transform=axes[row, col].transAxes,
                                        ha='right', va='top', fontsize=8)
            handles, labels = axes[row, 1].get_legend_handles_labels()
            if handles:
                axes[row, 1].legend(handles, labels, fontsize=7, loc='upper right', ncol=2 if args.all_points else 1)
        axes[0, 0].set_title('Entire available recording')
        axes[0, 1].set_title('Most recent {:g} seconds'.format(args.last_seconds))
        fig.suptitle(title + ' | preliminary downloaded data', fontsize=13)
        fig.text(.5, .01, 'Blue: highlighted point; gray: other selected points (unless all-points mode). '
                 'Gaps = unavailable/invalid data. Dotted times: 5 s baroreflex, 40 s ECM reference.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, .95))
        figures.append((title.replace(' ', '_') + '_' + str(start // 4 + 1), fig))


def beat_intervals(global_table):
    """Only explicit new_beat rising edges; no invented cycles from assumed HR."""
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


def cycle_values(table, j, a, b):
    t, y = table['t'], table['y'][:, j]
    dt = float(np.median(np.diff(t)))
    if a < t[0] - dt * .1 or b > t[-1] + dt * .1:
        return None
    # Include closing boundary for trapezoidal integration and peak/trough inspection.
    mask = (t >= a - dt * .1) & (t <= b + dt * .1)
    tt, yy = t[mask], y[mask]
    if len(tt) < 3 or not np.isfinite(yy).all():
        return None
    if tt[0] - a > dt * 1.1 or b - tt[-1] > dt * 1.1 or np.max(np.diff(tt)) > 1.6 * dt:
        return None
    return tt, yy


def write_beats(series, intervals, out, args, figures):
    records, summaries = [], []
    for field, table in series.items():
        for j, name in enumerate(table['names']):
            rows = []
            for bi, (a, b) in enumerate(intervals):
                vals = cycle_values(table, j, a, b)
                if vals is None:
                    continue
                t, v = vals
                mean = float(np.sum((v[1:] + v[:-1]) * .5 * np.diff(t)) / (t[-1] - t[0]))
                item = dict(field=field, column=name, beat_index=bi, start_s=a, end_s=b,
                            minimum=float(v.min()), maximum=float(v.max()), mean=mean)
                records.append(item)
                rows.append(item)
            # Require the latest N consecutive intervals available within this file, not any earlier good beats.
            eligible = [i for i, (_, b) in enumerate(intervals) if b <= table['t'][-1] + 1e-9]
            target = eligible[-args.last_beats:]
            tail = [r for r in rows if r['beat_index'] in target]
            if len(target) < args.last_beats or len(tail) != args.last_beats:
                continue
            scale = max(max(abs(r['minimum']), abs(r['maximum'])) for r in tail)
            s = dict(field=field, column=name, beats=len(tail), start_s=tail[0]['start_s'], end_s=tail[-1]['end_s'])
            for metric in ('minimum', 'maximum', 'mean'):
                v = [r[metric] for r in tail]
                s[metric + '_spread_percent_of_signal_scale'] = 100 * (max(v) - min(v)) / scale if scale > 1e-30 else ''
            s['normalization_scale'] = scale
            summaries.append(s)
    for filename, rows in [('beat_statistics.csv', records), ('last_beats_summary.csv', summaries)]:
        if rows:
            with (out / filename).open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    # Show peak and trough trends at one explicit point. No averaging across points.
    chosen = ['Ca_cytosol', 'active_stress', 'myofiber_passive', 'Sff_mesh',
              'ecm_instantaneous_fiber_stress', 'hs_length', 'heart_rate', 'baro_c',
              'arterioles_resistance', 'veins_compliance']
    plotrows = []
    for field in chosen:
        rr = [r for r in records if r['field'] == field and (r['column'] == str(args.focus) or r['column'] == field)]
        if rr:
            plotrows.append((field, rr))
    for start in range(0, len(plotrows), 4):
        chunk = plotrows[start:start + 4]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(11, 2.5 * len(chunk) + 1), squeeze=False)
        for ax, (field, rr) in zip(axes[:, 0], chunk):
            x = [r['end_s'] for r in rr]
            # Separate missing beats with NaNs, so a rejected cycle cannot look continuous.
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
            ax.set_ylabel(field + '\n(file units)', fontsize=9)
            ax.set_xlabel('Beat end time (s)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('Are successive beats becoming repeatable? | DOF {} / global quantities'.format(args.focus))
        fig.tight_layout(rect=(0, 0, 1, .95))
        figures.append(('Beat_trends_' + str(start // 4 + 1), fig))
    return len(records), len(summaries)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder')
    ap.add_argument('--dofs', nargs='+', type=int, default=DEFAULT_DOFS)
    ap.add_argument('--focus', type=int, default=1477)
    ap.add_argument('--all-points', action='store_true')
    ap.add_argument('--last-seconds', type=float, default=5)
    ap.add_argument('--last-beats', type=int, default=5)
    ap.add_argument('--end-time', type=float, help='Optional user-chosen cutoff in seconds; keeps earlier values only')
    ap.add_argument('--save-plots', action='store_true')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()
    if args.last_seconds <= 0 or args.last_beats < 2:
        ap.error('last-seconds must be positive and last-beats at least 2')
    if args.focus not in args.dofs:
        args.dofs = [args.focus] + args.dofs
    folder = choose_folder(args.folder)
    if (folder / 'sim_output').is_dir() and not (folder / 'data.csv').is_file():
        folder = folder / 'sim_output'
    if not folder.is_dir():
        raise SystemExit('Folder does not exist: {}'.format(folder))
    out = folder / ('trend_review_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    report = {'folder': str(folder), 'files': {}, 'missing_fields': [], 'errors': [],
              'policy': 'Finite zeros and spikes preserved. Bad cells are NaN gaps. Malformed rows excluded. '
                        'Stop at first duplicate/decreasing timestamp. No interpolation or inferred time/DOF mapping.',
              'point_warning': 'Selected DOF labels require the same geometry/global ordering as the selector; '
                               'these are candidate midwall points, not anatomically verified exact midwall.',
              'units': 'Values remain in original file units; no stress-measure or concentration conversion.'}
    metadata = None
    try:
        metadata = json.loads((folder / 'output_metadata.json').read_text(encoding='utf-8-sig'))
        report['metadata_complete'] = metadata.get('complete')
        print('Metadata complete flag:', metadata.get('complete'))
    except (OSError, ValueError) as exc:
        report['metadata_note'] = 'Metadata unavailable/unreadable: ' + str(exc)
    inventory = {p.name.lower(): p for p in folder.glob('*.csv')}
    series, global_table = {}, None

    def read_one(path, dofs):
        print('Reading {} ({:.1f} MB) ...'.format(path.name, path.stat().st_size / 1024**2), flush=True)
        try:
            key = 'global_rows_written' if dofs is None else 'spatial_rows_written'
            cap = metadata.get(key) if metadata else None
            cap = cap if isinstance(cap, int) and cap >= 0 else None
            tab = load_csv(path, dofs, max_rows=cap)
            if cap is not None:
                report['metadata_' + key] = cap
            if args.end_time is not None:
                keep = tab['t'] <= args.end_time
                tab['t'], tab['y'] = tab['t'][keep], tab['y'][keep]
            if len(tab['t']) < 2:
                raise ValueError('Not enough samples after requested cutoff')
            report['files'][path.name] = tab['report']
            print('  Plotting {:.3f} to {:.3f} s; {} selected columns'.format(tab['t'][0], tab['t'][-1], len(tab['names'])))
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
            tab = read_one(path, args.dofs)
            if tab is not None:
                series[field] = tab
        else:
            report['missing_fields'].append(field)
    if global_table is not None:
        for name in GLOBAL:
            v = get_trace(global_table, name)
            if v is not None:
                series[name] = {'t': global_table['t'], 'y': v[:, None], 'names': [name]}
        # RC indices may be MPI-local. Never identify them with selected global DOFs.
        for prefix, count in global_table['report'].get('rc_column_counts', {}).items():
            v = get_trace(global_table, prefix + 'recorded_mean')
            series[prefix + 'recorded_mean'] = {'t': global_table['t'], 'y': v[:, None],
                                                'names': ['mean of {} recorded RC columns'.format(count)]}
            report.setdefault('rc_coverage', {})[prefix] = {'count': count, 'note': 'Recorded columns only; global spatial coverage unverified. Means computed while streaming.'}
    if not series:
        (out / 'quality_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise SystemExit('No usable requested signals. Check folder/CSV headers. Report: {}'.format(out))
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figures = []
    titles = dict(SPATIAL, **GLOBAL)
    groups = [
        ('Calcium handling', ['Ca_cytosol', 'Ca_SR', 'J_release', 'J_uptake', 'Ca_content']),
        ('Contraction and relaxation', ['active_stress', 'hs_length', 'fiber_strain', 'fiber_stretch_raw']),
        ('Passive myofiber and ECM', ['Sff_mesh', 'myofiber_passive', 'ecm_instantaneous_fiber_stress']),
        ('Actual kinetic parameters', ['k_act', 'k_serca', 'k_1', 'k_on']),
        ('Baroreflex and circulation', ['heart_rate', 'baro_b', 'baro_c', 'arterioles_resistance', 'veins_compliance', 't_quiescent_period']),
        ('Pressure and volume', ['pressure_ventricle', 'pressure_aorta', 'volume_ventricle']),
        ('ECM reference diagnostics', ['ecm_baseline_stable', 'ecm_baseline_max_relative_change', 'ecm_reference_provisional']),
    ]
    for title, fields in groups:
        panels(title, [(titles[k], series[k]) for k in fields if k in series], args, figures)
    rc_items = [(p.replace('_rc_', '') + '\nRC recorded-column mean', series[p + 'recorded_mean'])
                for p in RC_PREFIXES if p + 'recorded_mean' in series]
    panels('Controller signals (RC): not actual kinetic rates', rc_items, args, figures)
    intervals = beat_intervals(global_table) if global_table is not None else []
    counts = write_beats(series, intervals, out, args, figures) if intervals else (0, 0)
    report['complete_beat_intervals_from_new_beat'] = intervals
    report['beat_statistics_rows'], report['last_beats_summary_rows'] = counts
    report['requested_end_time'] = args.end_time
    notes = [
        'PRELIMINARY REVIEW OF DOWNLOADED MYOFE OUTPUTS',
        'Input files were not modified. Each file retains its own actual time coverage.',
        'Blue highlights DOF {}; other selected points are gray (unless --all-points).'.format(args.focus),
        'Metadata saved-row counters, when present, cap reading conservatively; stale metadata can omit later valid downloaded rows.',
        'Zeros are kept: zero active/passive stress can be legitimate. All-zero signals are flagged in quality_report.json.',
        'NaN/Inf/non-numeric values are gaps. Malformed rows are excluded. Missing timestamps create gaps.',
        'First duplicate/decreasing timestamp ends that file: check report for possible padding or restarted recordings.',
        'Finite outliers are retained because a large value may be a real instability. Use --end-time S only for an explicit cutoff.',
        'All values use file units. Verify units and definitions before interpreting magnitude.',
        'Sff_mesh and myofiber_passive are shown separately; no assumption that they are equivalent stress measures.',
        'Selected DOFs require the same mesh/global ordering as the original selector.',
        'RC means describe recorded columns only; they are not asserted to represent the whole myocardium.',
        'For calcium/stress, repeatable beats matter; instantaneous signals should not become flat.',
        'Beat plots use explicit new_beat edges. First and last unbounded partial cycles are omitted.',
        'Any beat with invalid samples or time gaps in a trace is excluded for that trace.',
        'last_beats_summary.csv requires the latest {} consecutive complete intervals within each file.'.format(args.last_beats),
        'Spread = 100*(max statistic - min statistic)/largest absolute signal value across those beats.',
        'This normalization avoids division by a nearly zero trough. It is not a validated plateau threshold.',
        'Small spreads suggest repeatability of extrema/mean; waveform shape and timing still need inspection.',
        'No automatic pass/fail plateau or physiological-validity claim is made.',
        'Beat summaries unavailable: need explicit new_beat markers and enough valid complete beats.' if not counts[0] else 'Beat summaries created.',
        'Missing fields: ' + ', '.join(report['missing_fields']),
        'Errors: ' + '; '.join(report['errors']),
    ]
    (out / 'quality_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'review_notes.txt').write_text('\n'.join(notes), encoding='utf-8')
    if args.save_plots:
        for name, fig in figures:
            fig.savefig(out / (name + '.png'), dpi=150, bbox_inches='tight')
    print('\nREVIEW FOLDER:', out)
    print('Figures:', len(figures), '| beat-statistic rows:', counts[0])
    print('Missing fields:', ', '.join(report['missing_fields']) or 'none')
    print('Read quality_report.json for bad rows, zeros, missing DOFs, and actual coverage.')
    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
