#!/usr/bin/env python3
"""MyoFE_H sarcomere diagnostics: standard library only, large CSV safe.

Writes browser-viewable SVG plots and JSON. Does not need numpy/matplotlib.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

DEFAULT_DOFS = [577, 1473, 1477, 1941, 2589, 3041, 4629, 4633]
FIELDS = {
    'hs_length': ['hs_length_data.csv'],
    'fiber_strain': ['fiber_strain_data.csv'],
    'fiber_stretch_raw': ['fiber_stretch_raw_data.csv'],
    'active_stress': ['active_stress_data.csv'],
    'myofiber_passive': ['myofiber_passive_data.csv', 'Sff_mesh_data.csv'],
    'ecm_stress': ['ecm_instantaneous_fiber_stress_data.csv'],
    'Ca_cytosol': ['Ca_cytosol_data.csv'],
}


def finite(s):
    try:
        n = float(s)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def parse_dof(s):
    match = re.fullmatch(r'(?:dof[_ ]?|point[_ ]?|global_dof[_ ]?)?(\d+)', s.strip(), re.I)
    return int(match.group(1)) if match else None


def read_spatial(path, dofs, start, end, stride):
    """Keep one out of stride rows; never hold the entire source CSV in RAM."""
    points, found, count, previous = [], [], 0, -float('inf')
    zeros = [0] * len(dofs)
    negatives = [0] * len(dofs)
    with path.open('r', newline='', encoding='utf-8-sig', errors='replace') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ids = [parse_dof(x) for x in header]
        columns = [ids.index(d) if d in ids else None for d in dofs]
        if all(i is None for i in columns):
            raise ValueError('No requested DOF IDs in header: ' + ', '.join(header[:15]))
        times = [i for i, x in enumerate(header) if x.strip().lower() in ('time', 'time(s)', 'time_s', 't')]
        ti = times[0] if times else 0
        for row in reader:
            t = finite(row[ti]) if len(row) > ti else None
            if t is None:
                continue
            if t <= previous:
                print('WARNING: duplicate/decreasing timestamp in', path.name, 'at', t)
                break
            previous = t
            if t < start:
                continue
            if t > end:
                break
            vals = [finite(row[i]) if i is not None and i < len(row) else None for i in columns]
            for j, v in enumerate(vals):
                if v is not None:
                    zeros[j] += int(v == 0)
                    negatives[j] += int(v < 0)
            if count % stride == 0:
                points.append((t, vals))
            count += 1
    return points, {'file': str(path), 'available_dofs': [d for d, i in zip(dofs, columns) if i is not None],
                    'time_rows': count, 'plot_rows': len(points),
                    'zero_counts_by_dof': dict(zip(dofs, zeros)),
                    'negative_counts_by_dof': dict(zip(dofs, negatives))}


def svg_file(path, plots, title, xlabel='Time (s)', width=1100, panel_height=215):
    """plots: [(ylabel, [(label, [(x,y), ...], color)], ...)]"""
    height = 85 + len(plots) * panel_height + 50
    buf = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#253348} .grid{stroke:#dce4ed;stroke-width:1} .axis{stroke:#53657a;stroke-width:1.3}</style>',
           f'<text x="55" y="35" font-size="21" font-weight="bold">{escape(title)}</text>']
    left, right = 90, width - 35
    for k, (ylabel, series) in enumerate(plots):
        top = 65 + k * panel_height
        bottom = top + 155
        allxy = [(x, y) for _, xy, _ in series for x, y in xy if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
        if not allxy:
            continue
        xmin, xmax = min(x for x, _ in allxy), max(x for x, _ in allxy)
        ymin, ymax = min(y for _, y in allxy), max(y for _, y in allxy)
        dx, dy = xmax - xmin or 1., ymax - ymin or 1.
        ymin -= .04 * dy
        ymax += .04 * dy
        dy = ymax - ymin
        xp = lambda x: left + (x-xmin)/(dx) * (right-left)
        yp = lambda y: bottom - (y-ymin)/dy * (bottom-top)
        for i in range(5):
            gx = left + (right-left)*i/4
            gy = top + (bottom-top)*i/4
            buf.append(f'<line class="grid" x1="{gx:.1f}" x2="{gx:.1f}" y1="{top}" y2="{bottom}"/>')
            buf.append(f'<line class="grid" x1="{left}" x2="{right}" y1="{gy:.1f}" y2="{gy:.1f}"/>')
            buf.append(f'<text x="{gx:.1f}" y="{bottom+18}" text-anchor="middle" font-size="11">{xmin+dx*i/4:.3g}</text>')
            buf.append(f'<text x="{left-8}" y="{gy+4:.1f}" text-anchor="end" font-size="11">{ymax-dy*i/4:.3g}</text>')
        buf.append(f'<text x="{left}" y="{top-9}" font-size="14" font-weight="bold">{escape(ylabel)}</text>')
        for j, (label, xy, color) in enumerate(series):
            # Separate disjoint runs; never connect through missing values.
            run = []
            for x, y in xy + [(None, None)]:
                if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                    run.append(f'{xp(x):.1f},{yp(y):.1f}')
                elif run:
                    if len(run) > 1:
                        buf.append(f'<polyline points="{" ".join(run)}" fill="none" stroke="{color}" stroke-width="1.35"/>')
                    elif len(run) == 1:
                        px, py = run[0].split(',')
                        buf.append(f'<circle cx="{px}" cy="{py}" r="2" fill="{color}"/>')
                    run = []
            lx = left + j * 240
            buf.append(f'<line x1="{lx}" y1="{bottom+36}" x2="{lx+22}" y2="{bottom+36}" stroke="{color}" stroke-width="3"/>')
            buf.append(f'<text x="{lx+28}" y="{bottom+40}" font-size="12">{escape(label)}</text>')
    buf.append(f'<text x="{(left+right)/2:.1f}" y="{height-12}" text-anchor="middle" font-size="14">{escape(xlabel)}</text>')
    buf.append('</svg>')
    path.write_text('\n'.join(buf), encoding='utf-8')


def paired(a, b, index, tolerance):
    """Pair same-time saved outputs only; do not interpolate across gaps."""
    out = []
    j = 0
    for t, av in a:
        while j < len(b) and b[j][0] < t-tolerance:
            j += 1
        if j < len(b) and abs(b[j][0]-t) <= tolerance:
            x, y = av[index], b[j][1][index]
            if x is not None and y is not None:
                out.append((t, x, y))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', '--data-dir', dest='folder', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('sarcomere_diagnostics'))
    ap.add_argument('--start', type=float, default=50.)
    ap.add_argument('--end', type=float, default=60.)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--point', type=int, default=1477)
    ap.add_argument('--dofs', type=int, nargs='+', default=DEFAULT_DOFS)
    args = ap.parse_args()
    if args.start >= args.end or args.stride < 1 or args.point not in args.dofs:
        ap.error('Need --start < --end; --stride >= 1; --point included in --dofs')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data, quality = {}, {}
    for key, candidates in FIELDS.items():
        path = next((args.folder / x for x in candidates if (args.folder / x).is_file()), None)
        if path is None:
            print('MISSING:', key)
            continue
        data[key], quality[key] = read_spatial(path, args.dofs, args.start, args.end, args.stride)
        print(key, ':', quality[key]['time_rows'], 'rows in window')
    if not data:
        raise SystemExit('No files found. Check --folder path.')
    idx = args.dofs.index(args.point)
    panels = []
    for key, rows in data.items():
        point = [(t, v[idx]) for t, v in rows]
        average = [(t, sum(x for x in v if x is not None)/sum(x is not None for x in v)
                    if any(x is not None for x in v) else None) for t, v in rows]
        panels.append((key, [(f'DOF {args.point}', point, '#1768ac'),
                             ('8-point mean (unweighted)', average, '#d25435')]))
    svg_file(args.out_dir / '01_time_traces.svg', panels, 'MyoFE_H midwall sarcomere and stress traces')

    if 'myofiber_passive' in data and 'fiber_strain' in data:
        matches = paired(data['fiber_strain'], data['myofiber_passive'], idx, args.stride*.00051)
        xy = [(strain, xi) for _, strain, xi in matches]
        svg_file(args.out_dir / '02_Xi_vs_strain.svg',
                 [('Passive Xi stress', [('Xi versus strain', xy, '#1768ac')])],
                 f'Passive Xi vs fiber strain, DOF {args.point}', 'Fiber strain')
        quality['zero_Xi_test'] = {'paired_samples': len(matches),
            'Xi_zero_and_negative_strain': sum(xi == 0 and e < 0 for _, e, xi in matches),
            'Xi_positive_and_negative_strain': sum(xi > 0 and e < 0 for _, e, xi in matches),
            'Xi_zero_and_nonnegative_strain': sum(xi == 0 and e >= 0 for _, e, xi in matches),
            'Xi_negative_saved': sum(xi < 0 for _, e, xi in matches)}
    if 'hs_length' in data and 'active_stress' in data:
        matches = paired(data['hs_length'], data['active_stress'], idx, args.stride*.00051)
        xy = [(hsl, stress) for _, hsl, stress in matches]
        svg_file(args.out_dir / '03_active_stress_vs_HSL.svg',
                 [('Active stress', [('Active stress versus HSL', xy, '#16826d')])],
                 f'Active stress vs half-sarcomere length, DOF {args.point}',
                 'Half-sarcomere length (file units)')
    quality['disclaimer'] = ('An unweighted mean of selected DOFs is NOT a verified finite-element volume average. '
                             'Saved Xi=0 cannot prove negative raw stress or establish what was passed into MyoSim.')
    (args.out_dir / 'diagnostic_summary.json').write_text(json.dumps(quality, indent=2))
    print('DONE:', args.out_dir.resolve())


if __name__ == '__main__':
    main()
