#!/usr/bin/env python
"""Plot healthy versus AS volume-neutral fibrosis diagnostics."""
from __future__ import print_function
import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('healthy_csv')
    parser.add_argument('as_csv')
    parser.add_argument('--save-directory')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.headless:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = [('Healthy', pd.read_csv(args.healthy_csv)),
             ('Aortic stenosis', pd.read_csv(args.as_csv))]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for label, data in cases:
        time = data['time']
        axes[0, 0].plot(time,
            data['fib_cycle_averaged_total_fiber_stress_mean'], label=label)
        axes[0, 1].plot(time, data['fib_phi_c_mean'], label=label)
        axes[1, 0].plot(time, data['fib_phi_g_mean'], label=label)
        axes[1, 1].plot(time, data['fib_reference_volume'], label=label)
    axes[0, 0].set_ylabel('Mean cycle-averaged total fiber stress (model units)')
    axes[0, 1].set_ylabel('Mean collagen fraction (-)')
    axes[1, 0].set_ylabel('Mean ground-matrix fraction (-)')
    axes[1, 1].set_ylabel('Reference myocardial volume (mesh units^3)')
    for axis in axes.flat:
        axis.set_xlabel('Simulation/remodeling time (s)')
        axis.grid(True)
        axis.legend()
    fig.tight_layout()

    pv_fig, pv_axis = plt.subplots(figsize=(7, 6))
    for label, data in cases:
        if 'pressure_ventricle' not in data or 'volume_ventricle' not in data:
            raise KeyError('CSV lacks pressure_ventricle or volume_ventricle')
        event = data['fib_event_number']
        beat = (data['new_beat'] > 0).astype(int).cumsum()
        for event_value, suffix, style in (
                (event.min(), 'before', '--'), (event.max(), 'after', '-')):
            candidates = data[event == event_value]
            candidate_beats = beat[event == event_value]
            if len(candidates):
                selected_beat = candidate_beats.max()
                subset = data[beat == selected_beat]
                pv_axis.plot(subset['volume_ventricle'],
                             subset['pressure_ventricle'], style,
                             label='%s %s' % (label, suffix))
    pv_axis.set_xlabel('LV cavity volume (L, production circulation units)')
    pv_axis.set_ylabel('LV pressure (mmHg)')
    pv_axis.grid(True)
    pv_axis.legend()
    pv_fig.tight_layout()

    if args.save_directory:
        if not os.path.isdir(args.save_directory):
            os.makedirs(args.save_directory)
        fig.savefig(os.path.join(args.save_directory,
                                 'fibrosis_evolution.png'), dpi=200)
        pv_fig.savefig(os.path.join(args.save_directory,
                                    'fibrosis_pv_loops.png'), dpi=200)
    if not args.headless:
        plt.show()


if __name__ == '__main__':
    main()
