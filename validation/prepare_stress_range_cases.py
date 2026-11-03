#!/usr/bin/env python
"""Create runnable signed-range cases from matched Br inputs.

The stress limits are mandatory command-line measurements; this tool never
chooses or estimates scientific reference limits.
"""
from __future__ import print_function

import argparse
import json
import math
import os


def load(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def write(path, value):
    with open(path, 'w') as handle:
        json.dump(value, handle, indent=4, sort_keys=False)
        handle.write('\n')


def configure(case, stress_min, stress_max, result_root):
    fibrosis = case['model']['fibrosis']
    fibrosis['stimulus'] = ['cycle_averaged_total_fiber_stress_range']
    fibrosis['S_ff_min'] = [stress_min]
    fibrosis['S_ff_max'] = [stress_max]
    output = case['output_handler']
    output['output_data_path'] = [os.path.join(
        result_root, 'sim_output', 'data.csv')]
    output['mesh_output_path'] = [os.path.join(result_root, 'mesh_output')]
    fields = output['mesh_object_to_save']
    for name in ('fibrosis_stress_range_normalized', 'fibrosis_signal',
                 'fibrosis_S_ff_min', 'fibrosis_S_ff_max'):
        if name not in fields:
            fields.append(name)
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control_json')
    parser.add_argument('perturbed_json')
    parser.add_argument('S_ff_min', type=float)
    parser.add_argument('S_ff_max', type=float)
    parser.add_argument('output_directory')
    parser.add_argument('result_root')
    args = parser.parse_args()
    if not all(not math.isnan(v) and not math.isinf(v)
               for v in (args.S_ff_min, args.S_ff_max)):
        raise ValueError('Stress references must be finite')
    if args.S_ff_max <= args.S_ff_min:
        raise ValueError('S_ff_max must exceed S_ff_min')
    if not os.path.isdir(args.output_directory):
        os.makedirs(args.output_directory)
    paths = []
    for label, source in (('control', args.control_json),
                          ('perturbed', args.perturbed_json)):
        case = configure(load(source), args.S_ff_min, args.S_ff_max,
                         os.path.join(args.result_root, label))
        path = os.path.join(args.output_directory,
                            'stress_range_%s.json' % label)
        write(path, case)
        paths.append(path)
    print('\n'.join(paths))


if __name__ == '__main__':
    main()
