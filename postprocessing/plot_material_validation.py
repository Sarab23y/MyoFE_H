#!/usr/bin/env python
"""Plot biaxial or shear validation CSV files."""
from __future__ import print_function

import argparse
import glob
import os

import matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='CSV file or directory')
    parser.add_argument('--save', help='output image path')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.headless:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    files = ([args.input] if os.path.isfile(args.input) else
             sorted(glob.glob(os.path.join(args.input, '*.csv'))))
    if not files:
        raise ValueError('No validation CSV files found')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for path in files:
        data = pd.read_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]
        if 'prescribed_gamma' in data:
            x = data['prescribed_gamma']
            axes[0].plot(x, data['S_fs_total'], label=label)
            axes[1].plot(x, data['energy_total'], label=label)
            axes[0].set_xlabel('Engineering shear gamma [-]')
            axes[0].set_ylabel('Total S_fs [internal stress unit]')
            axes[1].set_xlabel('Engineering shear gamma [-]')
        else:
            axes[0].plot(data['E_ff'], data['S_ff_total'], label=label+' fiber')
            axes[0].plot(data['E_ss'], data['S_ss_total'], '--', label=label+' sheet')
            axes[1].plot(data['lambda_f'], data['energy_total'], label=label)
            axes[0].set_xlabel('Green-Lagrange strain [-]')
            axes[0].set_ylabel('Second Piola-Kirchhoff stress [internal unit]')
            axes[1].set_xlabel('Fiber stretch lambda_f [-]')
        axes[1].set_ylabel('Weighted passive energy [internal unit]')
    for axis in axes:
        axis.grid(True)
        axis.legend(fontsize=8)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=300)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
