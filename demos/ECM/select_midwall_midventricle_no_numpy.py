#!/usr/bin/env python3
"""Select candidate LV quadrature points using Python 3.6+ standard library only.

Reads numeric .npy geometry arrays, never modifies simulation files.
Wall depth is assumed normalized endo-to-epi; Z is the assumed long axis.
Verify these assumptions and selected positions on the mesh before interpretation.
"""
import argparse
import ast
import csv
import math
from pathlib import Path
import struct

DEFAULT_BASE = '/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_healthy_quad_validation_optimized/sim_output'


def read_npy(path):
    """Read real numeric NPY v1/v2/v3 without NumPy or pickle."""
    with open(str(path), 'rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('Not an NPY file: {}'.format(path))
        version = f.read(2)
        if len(version) != 2 or version[0] not in (1, 2, 3):
            raise ValueError('Unsupported NPY version: {}'.format(path))
        count = 2 if version[0] == 1 else 4
        raw = f.read(count)
        if len(raw) != count:
            raise ValueError('Truncated NPY header')
        header_size = struct.unpack('<H' if count == 2 else '<I', raw)[0]
        if header_size > 1024 * 1024:
            raise ValueError('Unexpectedly large NPY header')
        header = ast.literal_eval(f.read(header_size).decode('utf-8' if version[0] == 3 else 'latin1').strip())
        shape = header['shape']
        dtype = header['descr']
        formats = {'f4': 'f', 'f8': 'd', 'i1': 'b', 'i2': 'h', 'i4': 'i', 'i8': 'q',
                   'u1': 'B', 'u2': 'H', 'u4': 'I', 'u8': 'Q', 'b1': '?'}
        if not isinstance(dtype, str) or dtype[0] not in '<>|=' or dtype[1:] not in formats:
            raise ValueError('Unsupported numeric dtype {!r}; object arrays are never loaded'.format(dtype))
        if not isinstance(shape, tuple) or not shape or len(shape) > 2 or any(type(s) is not int or s < 0 for s in shape):
            raise ValueError('Expected a 1-D or 2-D numeric array')
        n = 1
        for s in shape:
            n *= s
        fmt = ('=' if dtype[0] == '|' else dtype[0]) + formats[dtype[1:]]
        itemsize = struct.calcsize(fmt)
        payload = f.read()
        if len(payload) != n * itemsize:
            raise ValueError('NPY size mismatch: {}'.format(path))
        values = [v[0] for v in struct.iter_unpack(fmt, payload)]
        if header['fortran_order'] and len(shape) == 2:
            rows, cols = shape
            values = [values[r + rows*c] for r in range(rows) for c in range(cols)]
        return shape, values


def save_rows(path, indices, ids, xyz, wall, zfrac):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['global_dof', 'x', 'y', 'z', 'wall_depth', 'z_fraction'])
        for i in indices:
            writer.writerow([ids[i]] + xyz[i] + [wall[i], zfrac[i]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sim-output', default=DEFAULT_BASE)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    base = Path(args.sim_output)
    out = Path(args.out) if args.out else base.parent / 'midwall_midventricle_selection_no_numpy'
    shape_xyz, coords = read_npy(base / 'quadrature_dof.npy')
    shape_ids, raw_ids = read_npy(base / 'quadrature_global_dof.npy')
    shape_wall, wall = read_npy(base / 'norm_dist_endo.npy')
    n = len(raw_ids)
    if n == 0 or shape_xyz != (n, 3) or len(wall) != n:
        raise ValueError('Inconsistent geometry shapes: {}, {}, {}'.format(shape_xyz, shape_ids, shape_wall))
    if any(not math.isfinite(v) for v in coords + wall + raw_ids):
        raise ValueError('Geometry contains nonfinite values')
    if any(v != int(v) or v < 0 for v in raw_ids):
        raise ValueError('Global DOF IDs must be nonnegative integers')
    ids = [int(v) for v in raw_ids]
    if len(set(ids)) != n:
        raise ValueError('Duplicate global DOF IDs')
    xyz = [coords[3*i:3*i+3] for i in range(n)]
    z = [p[2] for p in xyz]
    zmin, zmax = min(z), max(z)
    if zmax == zmin or min(wall) < 0 or max(wall) > 1:
        raise ValueError('Expected nonzero Z range and normalized wall depth within [0, 1]')
    zfrac = [(v-zmin)/(zmax-zmin) for v in z]
    score = [(wall[i]-.5)**2 + (zfrac[i]-.5)**2 for i in range(n)]
    print('Number of points:', n)
    print('Wall-depth range:', min(wall), max(wall))
    print('Z-coordinate range:', zmin, zmax)
    out.mkdir(parents=True, exist_ok=True)
    save_rows(out / 'all_point_geometry.csv', range(n), ids, xyz, wall, zfrac)
    closest = sorted(range(n), key=lambda i: (score[i], ids[i]))[:25]
    save_rows(out / 'closest_25_points.csv', closest, ids, xyz, wall, zfrac)

    pool = None
    band = None
    for low, high in ((.40, .60), (.35, .65), (.30, .70), (.25, .75)):
        candidates = [i for i in range(n) if low <= wall[i] <= high and low <= zfrac[i] <= high]
        print('Joint wall/Z band {:.2f}-{:.2f}: {} candidates'.format(low, high, len(candidates)))
        if len(candidates) >= 8:
            pool, band = candidates, (low, high)
            break
    if pool is None:
        print('No selection: fewer than eight candidates even in the 25-75% band.')
        print('Inspect closest_25_points.csv; these are NOT verified midwall points.')
        return

    ring = [i for i in range(n) if .35 <= zfrac[i] <= .65]
    cx = sum(xyz[i][0] for i in ring)/len(ring)
    cy = sum(xyz[i][1] for i in ring)/len(ring)
    # One closest-to-center point per occupied angular sector. Do not silently
    # fill empty sectors from another sector and claim circumferential coverage.
    sectors = [[] for _ in range(8)]
    for i in pool:
        angle = math.atan2(xyz[i][1]-cy, xyz[i][0]-cx) % (2*math.pi)
        sectors[min(7, int(angle/(2*math.pi)*8))].append(i)
    selected = []
    for sector, members in enumerate(sectors):
        if members:
            selected.append(min(members, key=lambda i: (score[i], ids[i])))
        else:
            print('NOTE: angular sector {} has no candidates; left empty.'.format(sector+1))
    save_rows(out / 'selected_midwall_midventricle_points.csv', selected, ids, xyz, wall, zfrac)
    (out / 'selected_ids.txt').write_text(' '.join(str(ids[i]) for i in selected) + '\n')
    print('\nSELECTED {} CANDIDATE POINTS using {:.0f}-{:.0f}% bands:'.format(len(selected), 100*band[0], 100*band[1]))
    for i in selected:
        print('  DOF {}  wall={:.4f}  zfrac={:.4f}  x/y/z=({:.6g}, {:.6g}, {:.6g})'.format(ids[i], wall[i], zfrac[i], *xyz[i]))
    print('DOF IDs:', ' '.join(str(ids[i]) for i in selected))
    print('Saved:', out / 'selected_midwall_midventricle_points.csv')
    print('Verify Z is the LV long axis, wall-depth meaning and geometry/DOF ordering, and locations on the mesh.')
    print('These are candidates in the reported band, not anatomically verified exact midwall points.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, SyntaxError, struct.error) as exc:
        raise SystemExit('ERROR: {}'.format(exc))
