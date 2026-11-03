#!/usr/bin/env python
"""Diagnose a stopped ECM baseline run from CSV output; no DOLFIN required."""
from __future__ import print_function, division
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python_codes'))
from LV_simulation.dependencies.ecm_stability import stability_results


def read_csv(path):
    with open(path, 'r') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('No rows in %s' % path)
    return rows


def number(row, name):
    value = row.get(name, '')
    if value in ('', None):
        raise ValueError('Missing column/value %s' % name)
    return float(value)


def time_weighted_mean(times, values):
    if len(times) < 2 or times[-1] <= times[0]:
        raise ValueError('A complete sampled cycle needs at least two increasing times')
    integral = sum(0.5*(values[i]+values[i-1])*(times[i]-times[i-1])
                   for i in range(1, len(times)))
    return float(integral/(times[-1]-times[0]))


def scalar_cycles(rows, activation_time, count):
    usable = [row for row in rows if number(row, 'time') < activation_time]
    starts = [i for i, row in enumerate(usable)
              if number(row, 'new_beat') > 0.5]
    cycles = []
    for left, right in zip(starts[:-1], starts[1:]):
        segment = usable[left:right+1]
        times = [number(row, 'time') for row in segment]
        cycles.append({'start': times[0], 'stop': times[-1],
                       'duration': times[-1]-times[0],
                       'pressure': time_weighted_mean(times, [number(r, 'pressure_ventricle') for r in segment]),
                       'volume': time_weighted_mean(times, [number(r, 'volume_ventricle') for r in segment]),
                       'heart_rate': time_weighted_mean(times, [number(r, 'heart_rate') for r in segment])})
    if len(cycles) < count:
        raise ValueError('Only %d complete cycles detectable from saved new_beat samples; need %d' %
                         (len(cycles), count))
    return cycles[-count:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_directory')
    parser.add_argument('--activation-time', type=float, default=20.0)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--tolerance', type=float, default=0.05)
    args = parser.parse_args()
    data_path = os.path.join(args.result_directory, 'sim_output', 'data.csv')
    rows = read_csv(data_path)
    cycles = scalar_cycles(rows, args.activation_time, args.cycles)
    print('Recovered scalar cycle means from %s:' % data_path)
    for index, cycle in enumerate(cycles):
        print('  %d: %s' % (index+1, json.dumps(cycle, sort_keys=True)))

    spatial = os.path.join(args.result_directory, 'mesh_output',
                           'ecm_cycle_averaged_fiber_stress_data.csv')
    # Some output layouts place spatial CSVs beside data.csv.
    alternative = os.path.join(args.result_directory, 'sim_output',
                               'ecm_cycle_averaged_fiber_stress_data.csv')
    if not os.path.isfile(spatial) and os.path.isfile(alternative):
        spatial = alternative
    if not os.path.isfile(spatial):
        print('ECM field check unavailable: no saved cycle-averaged spatial CSV found.')
        print('Scalar pressure/volume/heart-rate checks can be reconstructed; the physical FE L2 check cannot.')
        return 4

    print('Found %s, but existing CSV output does not store cycle identity/FE integration weights.' % spatial)
    print('A pointwise diagnostic can be explored, but the production physical L2 stability metric cannot be reconstructed exactly offline.')
    return 4


if __name__ == '__main__':
    sys.exit(main())
