#!/usr/bin/env python
"""Plot before/after quadrature-point collagen maps from spatial CSV output."""
from __future__ import print_function
import argparse
import os
import pandas as pd


def point_values(frame, row):
    columns = [name for name in frame.columns
               if name not in ('Unnamed: 0', 'time')]
    return frame.loc[row, columns].astype(float).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spatial_output_directory')
    parser.add_argument('--save')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.headless:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = args.spatial_output_directory
    phi = pd.read_csv(os.path.join(directory, 'fib_phi_c_data.csv'))
    x = pd.read_csv(os.path.join(directory, 'lx_data.csv'))
    z = pd.read_csv(os.path.join(directory, 'lz_data.csv'))
    rows = [0, len(phi)-1]
    values = [point_values(phi, row) for row in rows]
    vmin = min(value.min() for value in values)
    vmax = max(value.max() for value in values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, row, value, title in zip(
            axes, rows, values, ('Before fibrosis', 'After fibrosis')):
        scatter = axis.scatter(point_values(x, row), point_values(z, row),
                               c=value, s=9, vmin=vmin, vmax=vmax,
                               cmap='viridis')
        axis.set_title(title)
        axis.set_xlabel('X (mesh coordinate units)')
        axis.set_ylabel('Z (mesh coordinate units)')
        axis.set_aspect('equal', adjustable='box')
    fig.colorbar(scatter, ax=axes.ravel().tolist(),
                 label='Collagen fraction (-)')
    if args.save:
        fig.savefig(args.save, dpi=200, bbox_inches='tight')
    if not args.headless:
        plt.show()


if __name__ == '__main__':
    main()
