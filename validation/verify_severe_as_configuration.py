#!/usr/bin/env python
"""Preflight the matched healthy/severe-AS volume-neutral fibrosis inputs."""
from __future__ import print_function

import argparse
import json


def first(mapping, name):
    return mapping[name][0]


def load(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('healthy_json')
    parser.add_argument('perturbed_json')
    args = parser.parse_args()
    healthy = load(args.healthy_json)
    perturbed = load(args.perturbed_json)

    for label, case in (('healthy', healthy), ('perturbed', perturbed)):
        protocol = case['protocol']
        fibrosis = case['model']['fibrosis']
        passive = case['mesh']['forms_parameters']['passive_law_parameters']
        if first(protocol, 'time_step') != 0.001 or \
                first(protocol, 'no_of_time_steps') != 60000:
            raise RuntimeError('%s protocol is not 60 s at dt=0.001 s' % label)
        if 'growth' in case['model']:
            raise RuntimeError('%s unexpectedly enables geometric growth' % label)
        if first(fibrosis, 'geometric_growth_enabled') is not False:
            raise RuntimeError('%s fibrosis geometric growth is enabled' % label)
        expected = (0.700, 0.274, 0.026)
        actual = tuple(first(passive, name)
                       for name in ('phi_m', 'phi_g', 'phi_c'))
        if actual != expected:
            raise RuntimeError('%s has unexpected initial fractions' % label)
        for name, expected_value in (
                ('tau_c', 50), ('phi_c_max', 0.15), ('gamma_c', 1.5)):
            if first(fibrosis, name) != expected_value:
                raise RuntimeError('%s has unexpected %s' % (label, name))

    if 'perturbation' in healthy['protocol']:
        raise RuntimeError('Healthy case contains a perturbation')
    perturbations = perturbed['protocol'].get(
        'perturbation', {}).get('perturbations', [])
    if len(perturbations) != 1:
        raise RuntimeError('Perturbed case must contain exactly one perturbation')
    perturbation = perturbations[0]
    expected = {
        'level': 'circulation', 'variable': 'aorta_resistance',
        't_start_s': 30, 't_stop_s': 40, 'total_change': 170}
    actual = dict((name, first(perturbation, name)) for name in expected)
    if actual != expected:
        raise RuntimeError('Unexpected severe-AS perturbation: %r' % actual)
    aorta = [item for item in perturbed['model']['circulation']['compartments']
             if first(item, 'name') == 'aorta'][0]
    baseline = first(aorta, 'resistance')
    if baseline != 20:
        raise RuntimeError('Aortic resistance baseline is not 20')

    print('PASS: matched volume-neutral fibrosis configurations')
    print('healthy aortic resistance:', baseline)
    print('perturbed expected aortic resistance after ramp:',
          baseline + expected['total_change'])
    print('gamma_c:', first(perturbed['model']['fibrosis'], 'gamma_c'))


if __name__ == '__main__':
    main()
