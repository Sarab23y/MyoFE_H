#!/usr/bin/env python
"""Check global invariants for healthy and AS fibrosis validation runs."""
from __future__ import print_function
import argparse
import csv


def final_row(path):
    with open(path, 'r') as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows
                 if float(row.get('fib_event_number', 0.0)) > 0.0]
    if not completed:
        raise RuntimeError('%s contains no fibrosis remodeling event' % path)
    return completed[-1]


def value(row, name):
    return float(row[name])


def validate_invariants(label, row, tolerance):
    if abs(value(row, 'fib_reference_volume_residual')) > tolerance:
        raise RuntimeError('%s changed myocardial reference volume' % label)
    if abs(value(row, 'fib_total_mass_residual')) > tolerance:
        raise RuntimeError('%s changed total constituent mass' % label)
    for suffix in ('min', 'mean', 'max'):
        if abs(value(row, 'fib_phi_m_' + suffix) - 0.70) > tolerance:
            raise RuntimeError('%s changed myofiber fraction' % label)
    fraction_sum = (value(row, 'fib_phi_m_mean') +
                    value(row, 'fib_phi_g_mean') +
                    value(row, 'fib_phi_c_mean'))
    if abs(fraction_sum - 1.0) > tolerance:
        raise RuntimeError('%s mean fractions do not sum to one' % label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('healthy_csv')
    parser.add_argument('as_csv')
    parser.add_argument('--absolute-tolerance', type=float, default=1.0e-8)
    parser.add_argument('--healthy-collagen-tolerance', type=float,
                        default=5.0e-3)
    args = parser.parse_args()
    healthy = final_row(args.healthy_csv)
    stenosis = final_row(args.as_csv)
    validate_invariants('healthy', healthy, args.absolute_tolerance)
    validate_invariants('AS', stenosis, args.absolute_tolerance)
    healthy_phi = value(healthy, 'fib_phi_c_mean')
    as_phi = value(stenosis, 'fib_phi_c_mean')
    if abs(healthy_phi - 0.03) > args.healthy_collagen_tolerance:
        raise RuntimeError('Healthy collagen accumulation is not negligible')
    if as_phi <= healthy_phi:
        raise RuntimeError('AS did not increase mean collagen over healthy')
    print('PASS: volume-neutral AS fibrosis invariants')
    print('healthy/AS final mean phi_c:', healthy_phi, as_phi)


if __name__ == '__main__':
    main()
