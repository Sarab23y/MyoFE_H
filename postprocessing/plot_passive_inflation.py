#!/usr/bin/env python
"""Plot a converged passive LV inflation pressure-volume curve."""
from __future__ import print_function

import argparse

import matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--save')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.headless:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.read_csv(args.csv)
    if data['converged'].dtype == object:
        converged_mask = data['converged'].astype(str).str.lower() == 'true'
    else:
        converged_mask = data['converged'].astype(bool)
    converged = data[converged_mask]
    if converged.empty:
        raise ValueError('No converged inflation states in CSV')
    if 'cavity_volume_ml' not in converged:
        raise ValueError('CSV has no cavity_volume_ml column')
    volume_ml = converged['cavity_volume_ml']
    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.plot(volume_ml, converged['pressure_mmHg'], 'o-')
    axis.set_xlabel('LV cavity volume [mL]')
    axis.set_ylabel('Applied cavity pressure [mmHg]')
    axis.set_title('Passive LV inflation pressure-volume curve')
    axis.grid(True)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=300)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
