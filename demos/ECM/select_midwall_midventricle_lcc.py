#!/usr/bin/env python3
"""Read-only MyoFE_H Gauss-point region selector with geometry diagnostics.

Z is used as the assumed LV long axis; verify anatomically with the mesh.
The script will NOT label fallback candidates as midwall if the data do not
support a reasonable wall depth and longitudinal location.
"""
import csv
from pathlib import Path
import numpy as np

BASE = Path('/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_healthy_quad_validation_optimized/sim_output')
OUT = BASE.parent / 'midwall_midventricle_selection_checked'
OUT.mkdir(exist_ok=True)

xyz = np.asarray(np.load(BASE / 'quadrature_dof.npy'), dtype=float)
gid = np.asarray(np.load(BASE / 'quadrature_global_dof.npy')).reshape(-1).astype(int)
wall = np.asarray(np.load(BASE / 'norm_dist_endo.npy'), dtype=float).reshape(-1)

n = len(gid)
if xyz.shape != (n, 3) or wall.shape != (n,):
    raise ValueError('Inconsistent shapes: xyz %s, IDs %s, wall %s' % (xyz.shape, gid.shape, wall.shape))
if np.unique(gid).size != n or not np.isfinite(xyz).all() or not np.isfinite(wall).all():
    raise ValueError('Duplicate DOFs or nonfinite values in geometry files.')

# DOF IDs and coordinates share global-DOF order in MyoFE buffered outputs.
z = xyz[:, 2]
if np.ptp(z) == 0:
    raise ValueError('Z has zero range; long-axis assumption is invalid.')
zfrac = (z - np.min(z)) / np.ptp(z)

print('\nGAUSS POINTS:', n)
print('X, Y, Z min:', xyz.min(axis=0))
print('X, Y, Z max:', xyz.max(axis=0))
print('Wall-depth min / max:', wall.min(), wall.max())
print('Wall-depth quantiles (0,10,25,50,75,90,100%):',
      np.round(np.percentile(wall, [0, 10, 25, 50, 75, 90, 100]), 4))
print('Wall-depth unique values (rounded to 3 decimals):', np.unique(wall.round(3)).size)
print('Z quantiles:', np.round(np.percentile(z, [0, 10, 25, 50, 75, 90, 100]), 4))
print('Near-midwall count (0.4 to 0.6):', int(np.count_nonzero((wall >= .4) & (wall <= .6))))
print('Mid-Z count (0.4 to 0.6):', int(np.count_nonzero((zfrac >= .4) & (zfrac <= .6))))

print('\nWall-depth histogram (all points):')
for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
    count = int(np.count_nonzero((wall >= low) & ((wall < high) if high < 1 else (wall <= high))))
    print('  %.1f–%.1f: %d' % (low, high, count))

with (OUT / 'all_point_geometry.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['global_dof', 'x', 'y', 'z', 'wall_depth', 'z_fraction'])
    for i in range(n):
        writer.writerow([gid[i], *xyz[i], wall[i], zfrac[i]])

if np.min(wall) < -.05 or np.max(wall) > 1.05:
    print('\nWARNING: wall-depth values fall outside the expected 0–1 range.')
    print('Do not treat them as normalized wall position until their definition is verified.')
    raise SystemExit(2)

# Expand in documented steps ONLY if the actual data support a central region.
# A width of .25 means 25%-75%, so never silently choose the wall surfaces.
selected_pool = None
chosen_width = None
for halfwidth in (.10, .15, .20, .25):
    mask = (np.abs(wall - .5) <= halfwidth) & (np.abs(zfrac - .5) <= halfwidth)
    candidates = np.flatnonzero(mask)
    print('Joint band [%.2f, %.2f] for wall and normalized Z: %d points' %
          (.5-halfwidth, .5+halfwidth, len(candidates)))
    if len(candidates) >= 8:
        selected_pool, chosen_width = candidates, halfwidth
        break

# Always report which points are geometrically closest to the target.
score = (wall - .5)**2 + (zfrac - .5)**2
closest = np.argsort(score)[:25]
with (OUT / 'closest_25_points.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['global_dof', 'x', 'y', 'z', 'wall_depth', 'z_fraction', 'distance_to_center'])
    for i in closest:
        writer.writerow([gid[i], *xyz[i], wall[i], zfrac[i], np.sqrt(score[i])])

if selected_pool is None:
    print('\nNO VERIFIED MIDWALL/MID-LV SELECTION under even the 25–75% bands.')
    print('Closest 10 points (NOT necessarily midwall):')
    for i in closest[:10]:
        print('  DOF %-5d wall=%.4f  zfrac=%.4f  x/y/z=%s' %
              (gid[i], wall[i], zfrac[i], np.round(xyz[i], 4)))
    print('Saved:', OUT / 'all_point_geometry.csv')
    print('Saved:', OUT / 'closest_25_points.csv')
    print('Please share the diagnostic output. Do not plot points as midwall yet.')
    raise SystemExit(0)

# Distribute points around the horizontal plane rather than selecting a cluster.
# Center estimated from the LV section near its longitudinal midpoint.
ring = np.abs(zfrac - .5) <= .15
center_xy = np.mean(xyz[ring, :2], axis=0)
angles = np.mod(np.arctan2(xyz[:, 1]-center_xy[1], xyz[:, 0]-center_xy[0]), 2*np.pi)
selected = []
for sector in range(8):
    target_angle = 2*np.pi*(sector + .5)/8
    d = np.abs(np.angle(np.exp(1j*(angles[selected_pool]-target_angle))))/np.pi
    ranking = 2*score[selected_pool] + .10*d*d
    for j in np.argsort(ranking):
        i = int(selected_pool[j])
        if i not in selected:
            selected.append(i)
            break

with (OUT / 'selected_midwall_midventricle_points.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['global_dof', 'x', 'y', 'z', 'wall_depth', 'z_fraction'])
    for i in selected:
        writer.writerow([gid[i], *xyz[i], wall[i], zfrac[i]])
(OUT / 'selected_ids.txt').write_text(' '.join(str(gid[i]) for i in selected) + '\n')

print('\nSELECTED %d POINTS using %.0f–%.0f%% bands (wall and Z):' %
      (len(selected), 100*(.5-chosen_width), 100*(.5+chosen_width)))
for i in selected:
    print('  DOF %-5d wall=%.4f  zfrac=%.4f  x/y/z=%s' %
          (gid[i], wall[i], zfrac[i], np.round(xyz[i], 4)))
print('DOF IDs:', ' '.join(str(gid[i]) for i in selected))
print('Saved:', OUT / 'selected_midwall_midventricle_points.csv')
print('IMPORTANT: Verify that Z is the true apex-to-base axis and confirm positions in ParaView.')
