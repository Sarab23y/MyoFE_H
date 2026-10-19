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

    fig, axes = plt.subplots(5, 2, figsize=(12, 17))
    for label, data in cases:
        time = data['time']
        axes[0, 0].plot(time,
            data['fib_cycle_averaged_total_fiber_stress_mean'], label=label)
        axes[0, 1].plot(time,
            data['fib_homeostatic_fiber_stress_mean'], label=label)
        normalized_name = ('fib_stress_range_normalized_mean'
            if 'fib_stress_range_normalized_mean' in data else
            'fib_normalized_stimulus_mean')
        axes[1, 0].plot(time, data[normalized_name],
                        label=label + ' normalized r')
        signal_name = ('fib_signal_mean' if 'fib_signal_mean' in data else
                       'fib_effective_stimulus_mean')
        axes[1, 0].plot(time, data[signal_name], '--',
                        label=label + ' final signal')
        axes[1, 1].plot(time, data['fib_phi_c_mean'], label=label)
        axes[2, 0].plot(time, data['fib_phi_g_mean'], label=label)
        axes[2, 1].plot(time, data['fib_reference_volume'], label=label)
        axes[3, 0].plot(time, data['aorta_resistance'], label=label)
        axes[3, 1].plot(time, data['fib_phi_c_max'], label=label)
        if 'fib_lower_bound_fraction' in data:
            axes[4, 0].plot(time, data['fib_lower_bound_fraction'],
                            label=label)
        if 'fib_upper_bound_fraction' in data:
            axes[4, 1].plot(time, data['fib_upper_bound_fraction'],
                            label=label)
    axes[0, 0].set_ylabel('Cycle-averaged total fiber stress (model units)')
    axes[0, 1].set_ylabel('Homeostatic fiber stress (model units)')
    axes[1, 0].set_ylabel('Normalized r (-) / final signal (1/s)')
    axes[1, 1].set_ylabel('Mean collagen fraction (-)')
    axes[2, 0].set_ylabel('Mean ground-matrix fraction (-)')
    axes[2, 1].set_ylabel('Reference myocardial volume (mesh units^3)')
    axes[3, 0].set_ylabel('Aortic resistance (model units)')
    axes[3, 1].set_ylabel('Maximum local collagen fraction (-)')
    axes[4, 0].set_ylabel('Material points at lower bound (-)')
    axes[4, 1].set_ylabel('Material points at upper bound (-)')
    for axis in axes.flat:
        axis.set_xlabel('Simulation/remodeling time (s)')
        axis.grid(True)
        axis.legend()
    fig.tight_layout()

    pv_fig, pv_axis = plt.subplots(figsize=(7, 6))
    def complete_beat(data, event_value):
        event = data['fib_event_number']
        beat = (data['new_beat'] > 0).astype(int).cumsum()
        candidate_beats = beat[event == event_value]
        if not len(candidate_beats):
            return data.iloc[0:0]
        return data[beat == candidate_beats.max()]

    healthy = cases[0][1]
    stenosis = cases[1][1]
    selections = (
        ('Healthy control', complete_beat(
            healthy, healthy['fib_event_number'].max()), '-'),
        ('Early AS', complete_beat(
            stenosis, stenosis['fib_event_number'].min()), '--'),
        ('Late AS/fibrosis', complete_beat(
            stenosis, stenosis['fib_event_number'].max()), '-'))
    for label, subset, style in selections:
        if 'pressure_ventricle' not in subset or \
                'volume_ventricle' not in subset:
            raise KeyError('CSV lacks pressure_ventricle or volume_ventricle')
        if len(subset):
            pv_axis.plot(subset['volume_ventricle'],
                         subset['pressure_ventricle'], style, label=label)

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
