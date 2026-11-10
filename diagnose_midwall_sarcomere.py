#!/usr/bin/env python3
"""Memory-light MyoFE_H point/region diagnostics (read-only)."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOFS = [577, 1473, 1477, 1941, 2589, 3041, 4629, 4633]
FIELDS = {
    'hs_length': ['hs_length_data.csv'],
    'fiber_strain': ['fiber_strain_data.csv'],
    'fiber_stretch_raw': ['fiber_stretch_raw_data.csv'],
    'active_stress': ['active_stress_data.csv'],
    'myofiber_passive': ['myofiber_passive_data.csv', 'Sff_mesh_data.csv'],
    'ecm_stress': ['ecm_instantaneous_fiber_stress_data.csv'],
    'Ca_cytosol': ['Ca_cytosol_data.csv'],
    'new_beat': ['new_beat_data.csv'],
}


def find_file(folder, choices):
    for name in choices:
        p = folder / name
        if p.exists():
            return p
    return None


def stream_field(path, dofs, start, end, stride, max_rows):
    """Read only matching columns, retain at most max_rows plot samples."""
    times, values = [], []
    nvalid = 0
    nzero = np.zeros(len(dofs), dtype=int)
    nnegative = np.zeros(len(dofs), dtype=int)
    nseen = np.zeros(len(dofs), dtype=int)
    with path.open('r', newline='', errors='replace') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        normalized = [x.strip().strip('"') for x in header]
        tindex = next((i for i, h in enumerate(normalized) if h.lower() in
                       ('time', 'time(s)', 'time_s', 't')), 0)
        column_ids = {d: next((i for i, h in enumerate(normalized) if h == str(d)), None)
                      for d in dofs}
        if all(i is None for i in column_ids.values()):
            raise ValueError(f'No selected DOF columns found in {path.name}: {normalized[:12]}')
        last_t = -float('inf')
        for row in reader:
            try:
                t = float(row[tindex])
            except (ValueError, IndexError):
                continue
            if not math.isfinite(t):
                continue
            if t <= last_t:
                print(f'WARNING {path.name}: time reset/duplicate at {t}; stopping')
                break
            last_t = t
            if t < start:
                continue
            if t > end:
                break
            v = np.full(len(dofs), np.nan)
            for j, d in enumerate(dofs):
                i = column_ids[d]
                if i is None or i >= len(row):
                    continue
                try:
                    v[j] = float(row[i])
                except ValueError:
                    pass
            ok = np.isfinite(v)
            nseen += ok.astype(int)
            nzero += (ok & (v == 0)).astype(int)
            nnegative += (ok & (v < 0)).astype(int)
            if nvalid % stride == 0:
                times.append(t)
                values.append(v)
            nvalid += 1
    if nvalid > max_rows * stride:
        print(f'WARNING {path.name}: {nvalid // stride} plot rows; reduce --max-points via larger --stride')
    return np.asarray(times), np.asarray(values).reshape(-1, len(dofs)), {
        'file': str(path), 'valid_time_rows': nvalid,
        'sampled_rows': len(times), 'zero_counts': nzero.tolist(),
        'negative_counts': nnegative.tolist(), 'finite_counts': nseen.tolist(),
        'missing_dofs': [d for d in dofs if column_ids[d] is None],
    }


def plot_panel(ax, data, field, dofs, point, regional):
    t, a = data[field]
    if a.size == 0:
        return
    if point in dofs:
        ax.plot(t, a[:, dofs.index(point)], color='#1768ac', lw=1.0,
                label=f'DOF {point}')
    if regional:
        with np.errstate(all='ignore'):
            mean = np.nanmean(a, axis=1)
            lo = np.nanmin(a, axis=1)
            hi = np.nanmax(a, axis=1)
        ax.plot(t, mean, color='#cf4a27', lw=1, alpha=.9, label='8-point mean')
        ax.fill_between(t, lo, hi, color='#cf4a27', alpha=.13,
                        label='Point range')
    ax.set_ylabel(field.replace('_', ' '))
    ax.grid(alpha=.25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True, help='Folder with *_data.csv')
    p.add_argument('--out-dir', type=Path, default=Path('sarcomere_diagnostics'))
    p.add_argument('--start', type=float, default=50., help='Start time in seconds')
    p.add_argument('--end', type=float, default=60., help='End time in seconds')
    p.add_argument('--stride', type=int, default=10, help='Keep every Nth time sample')
    p.add_argument('--max-points', type=int, default=10000,
                   help='Recommended maximum plot samples per signal')
    p.add_argument('--point', type=int, default=1477)
    p.add_argument('--dofs', type=int, nargs='+', default=DOFS)
    args = p.parse_args()
    if args.start >= args.end or args.stride < 1 or args.point not in args.dofs:
        p.error('Require start < end, stride >= 1 and --point among --dofs')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data, quality = {}, {}
    for field, alternatives in FIELDS.items():
        filename = find_file(args.data_dir, alternatives)
        if not filename:
            print(f'MISSING {field}: tried {alternatives}')
            continue
        try:
            t, a, q = stream_field(filename, args.dofs, args.start, args.end,
                                   args.stride, args.max_points)
            data[field] = (t, a)
            quality[field] = q
            print(f'{field}: {q["valid_time_rows"]} original rows, {len(t)} plotted')
        except (OSError, ValueError) as exc:
            print(f'ERROR {field}: {exc}')

    panels = [f for f in ['hs_length', 'fiber_strain', 'fiber_stretch_raw',
                          'active_stress', 'myofiber_passive', 'ecm_stress',
                          'Ca_cytosol'] if f in data]
    if not panels:
        raise SystemExit('No readable field files found. Check --data-dir and filenames.')
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2.45 * len(panels)),
                             sharex=True, constrained_layout=True)
    for ax, field in zip(np.atleast_1d(axes), panels):
        plot_panel(ax, data, field, args.dofs, args.point, True)
    axes[0].legend(loc='best', fontsize=8)
    axes[-1].set_xlabel('Simulation time (s)')
    fig.suptitle('Midwall sarcomere / passive-stress diagnostics (unweighted DOF mean)')
    fig.savefig(args.out_dir / '01_time_traces.png', dpi=160)
    plt.close(fig)

    # Compare Xi zeros against signed fiber strain, at the same selected point.
    if 'myofiber_passive' in data and 'fiber_strain' in data:
        tx, sx = data['myofiber_passive']
        te, se = data['fiber_strain']
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        xi = sx[:, args.dofs.index(args.point)]
        strain = np.interp(tx, te, se[:, args.dofs.index(args.point)],
                           left=np.nan, right=np.nan)
        valid = np.isfinite(strain) & np.isfinite(xi)
        ax.scatter(strain[valid], xi[valid], s=7, alpha=.35, color='#1768ac')
        ax.axvline(0, color='black', ls='--', lw=.8)
        ax.axhline(0, color='black', ls='--', lw=.8)
        ax.set(xlabel='Fiber strain (saved definition)',
               ylabel='Saved Xi passive stress (file units)',
               title=f'Xi versus strain, DOF {args.point}')
        ax.grid(alpha=.25)
        fig.savefig(args.out_dir / '02_Xi_vs_strain.png', dpi=180)
        plt.close(fig)
        quality['xi_vs_strain_point'] = {
            'point': args.point,
            'n_paired_samples': int(valid.sum()),
            'zero_Xi_and_negative_strain': int(np.sum(valid & (xi == 0) & (strain < 0))),
            'positive_Xi_and_negative_strain': int(np.sum(valid & (xi > 0) & (strain < 0))),
            'zero_Xi_and_nonnegative_strain': int(np.sum(valid & (xi == 0) & (strain >= 0))),
            'negative_Xi_saved': int(np.sum(valid & (xi < 0))),
            'note': 'Zeros alone cannot reveal whether a pre-clipped Xi stress was negative or whether clipping occurred inside MyoSim feedback.'
        }

    # Beat-level scatter: start with raw point values, not average-only curves.
    if 'hs_length' in data and 'active_stress' in data:
        th, hh = data['hs_length']
        ta, aa = data['active_stress']
        h = hh[:, args.dofs.index(args.point)]
        a = np.interp(th, ta, aa[:, args.dofs.index(args.point)],
                      left=np.nan, right=np.nan)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        mask = np.isfinite(h) & np.isfinite(a)
        sc = ax.scatter(h[mask], a[mask], c=th[mask], s=8, cmap='viridis')
        fig.colorbar(sc, ax=ax, label='Time (s)')
        ax.set(xlabel='Half-sarcomere length (file units)',
               ylabel='Active stress (file units)',
               title=f'Active stress vs HSL, DOF {args.point}')
        ax.grid(alpha=.25)
        fig.savefig(args.out_dir / '03_active_stress_vs_HSL.png', dpi=180)
        plt.close(fig)

    with (args.out_dir / 'diagnostic_summary.json').open('w') as fh:
        json.dump({'start': args.start, 'end': args.end, 'stride': args.stride,
                   'dofs': args.dofs, 'files': quality}, fh, indent=2)
    print(f'Wrote PNGs and diagnostic_summary.json to {args.out_dir.resolve()}')
    print('IMPORTANT: Selected DOFs are NOT proven to be a true control volume; averages are unweighted.')
    print('IMPORTANT: Saved Xi zeros cannot identify where the zeroing occurred in the solver.')


if __name__ == '__main__':
    main()
