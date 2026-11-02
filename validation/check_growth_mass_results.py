#!/usr/bin/env python
"""Check global invariants in the three controlled growth CSV files."""
from __future__ import print_function

import argparse
import csv


def last_event(path):
    with open(path, 'r') as handle:
        rows = list(csv.DictReader(handle))
    event_rows = [row for row in rows
                  if float(row.get('gr_growth_event_number', 0.0)) > 0.0]
    if not event_rows:
        raise RuntimeError('%s contains no completed growth event' % path)
    return event_rows[-1]


def number(row, key):
    return float(row[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('no_growth_csv')
    parser.add_argument('proportional_csv')
    parser.add_argument('collagen_enriched_csv')
    parser.add_argument('--relative-tolerance', type=float, default=1.0e-6)
    args = parser.parse_args()
    no_growth = last_event(args.no_growth_csv)
    proportional = last_event(args.proportional_csv)
    enriched = last_event(args.collagen_enriched_csv)
    tol = args.relative_tolerance

    for label, row in (('no growth', no_growth),
                       ('proportional', proportional),
                       ('collagen enriched', enriched)):
        mass = number(row, 'gr_global_mass_total')
        residual = number(row, 'gr_mass_balance_residual')
        if abs(residual) > tol*max(1.0, abs(mass)):
            raise RuntimeError('%s mass residual exceeds tolerance' % label)

    pre = number(no_growth, 'gr_pre_growth_myocardial_volume')
    post = number(no_growth, 'gr_post_growth_myocardial_volume')
    if abs(post - pre) > tol*max(1.0, abs(pre)):
        raise RuntimeError('No-growth reference volume changed')

    proportional_c = number(proportional, 'gr_global_mass_collagen') / \
        number(proportional, 'gr_global_mass_total')
    enriched_c = number(enriched, 'gr_global_mass_collagen') / \
        number(enriched, 'gr_global_mass_total')
    if abs(proportional_c - 0.03) > tol:
        raise RuntimeError('Proportional deposition did not preserve phi_c')
    if enriched_c <= proportional_c:
        raise RuntimeError('Collagen-enriched deposition did not raise phi_c')
    print('PASS: controlled growth mass checks')
    print('proportional/enriched global collagen fractions:',
          proportional_c, enriched_c)


if __name__ == '__main__':
    main()
