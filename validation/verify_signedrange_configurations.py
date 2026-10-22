#!/usr/bin/env python
"""Validate matched Br signed-range inputs and block unresolved bounds."""
from __future__ import print_function

import argparse
import json
import math


def first(mapping, name):
    value = mapping[name]
    if not isinstance(value, list) or not value:
        raise RuntimeError('%s must be a non-empty list' % name)
    return value[0]


def load(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control_json')
    parser.add_argument('perturbed_json')
    parser.add_argument('--allow-unresolved-bounds', action='store_true')
    args = parser.parse_args()
    cases = [('control', load(args.control_json)),
             ('perturbed', load(args.perturbed_json))]
    unresolved = False
    references = []
    for label, case in cases:
        protocol = case['protocol']
        fibrosis = case['model']['fibrosis']
        passive = case['mesh']['forms_parameters']['passive_law_parameters']
        if first(fibrosis, 'stimulus') != \
                'cycle_averaged_total_fiber_stress_range':
            raise RuntimeError('%s does not select signed-range fibrosis' % label)
        if first(fibrosis, 'gamma_c') != 1.5 or \
                first(fibrosis, 'tau_c') != 50 or \
                first(fibrosis, 'phi_c_max') != 0.15:
            raise RuntimeError('%s has incorrect fibrosis kinetics' % label)
        if first(fibrosis, 'geometric_growth_enabled') is not False or \
                'growth' in case['model']:
            raise RuntimeError('%s enables geometric growth' % label)
        fractions = tuple(first(passive, key)
                          for key in ('phi_m', 'phi_g', 'phi_c'))
        if fractions != (0.700, 0.274, 0.026):
            raise RuntimeError('%s has incorrect initial fractions' % label)
        if first(protocol, 'time_step') != 0.001 or \
                first(protocol, 'no_of_time_steps') != 60000:
            raise RuntimeError('%s is not a 60-second dt=0.001 case' % label)
        growth_activation = protocol['growth']['activations'][0]
        baro_activation = protocol['baroreflex']['activations'][0]
        if (first(growth_activation, 't_start_s'),
                first(growth_activation, 't_stop_s')) != (20, 60):
            raise RuntimeError('%s fibrosis window is incorrect' % label)
        if (first(baro_activation, 't_start_s'),
                first(baro_activation, 't_stop_s')) != (5, 60):
            raise RuntimeError('%s baroreflex window is incorrect' % label)
        lower = first(fibrosis, 'S_ff_min')
        upper = first(fibrosis, 'S_ff_max')
        if lower is None or upper is None:
            unresolved = True
        else:
            lower, upper = float(lower), float(upper)
            if math.isnan(lower) or math.isinf(lower) or \
                    math.isnan(upper) or math.isinf(upper) or upper <= lower:
                raise RuntimeError('%s stress references are invalid' % label)
            references.append((lower, upper))
    if 'perturbation' in cases[0][1]['protocol']:
        raise RuntimeError('Control unexpectedly contains a perturbation')
    changes = cases[1][1]['protocol']['perturbation']['perturbations']
    if len(changes) != 1:
        raise RuntimeError('Perturbed case needs exactly one perturbation')
    change = changes[0]
    actual = (first(change, 'level'), first(change, 'variable'),
              first(change, 't_start_s'), first(change, 't_stop_s'),
              first(change, 'total_change'))
    if actual != ('circulation', 'aorta_resistance', 30, 40, 170):
        raise RuntimeError('Incorrect severe-AS-like perturbation: %r' %
                           (actual,))
    if len(references) == 2 and references[0] != references[1]:
        raise RuntimeError('Control and perturbed references differ')
    if unresolved:
        message = ('BLOCKED: S_ff_min/S_ff_max require verified baseline '
                   'cycle-averaged fiber-stress data')
        print(message)
        if not args.allow_unresolved_bounds:
            raise RuntimeError(message)
    else:
        print('PASS: runnable matched signed-range configurations')
        print('S_ff_min/S_ff_max:', references[0])
    print('PASS: control has no perturbation')
    print('PASS: perturbed case applies aorta_resistance +170 from 30 to 40 s')


if __name__ == '__main__':
    main()
