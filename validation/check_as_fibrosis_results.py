#!/usr/bin/env python
"""Check global invariants for healthy and AS fibrosis validation runs."""
from __future__ import print_function
import argparse
import csv
import math


def final_row(path):
    with open(path, 'r') as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows
                 if float(row.get('fib_event_number', 0.0)) > 0.0]
    if not completed:
        raise RuntimeError('%s contains no fibrosis remodeling event' % path)
    return completed[-1]


def observed_range(path, minimum_name, maximum_name):
    with open(path, 'r') as handle:
        rows = list(csv.DictReader(handle))
    return (min(float(row[minimum_name]) for row in rows),
            max(float(row[maximum_name]) for row in rows))


def rows_from(path):
    with open(path, 'r') as handle:
        return list(csv.DictReader(handle))


def verify_aortic_resistance(rows, perturbed, tolerance):
    """Verify the saved scalar is the live circulation resistance."""
    before = [float(row['aorta_resistance']) for row in rows
              if float(row['time']) < 30.0]
    after = [float(row['aorta_resistance']) for row in rows
             if float(row['time']) >= 40.0]
    if not before or not after:
        raise RuntimeError('CSV does not span the 30--40 s AS ramp')
    if max(abs(value - 20.0) for value in before) > tolerance:
        raise RuntimeError('Pre-ramp aortic resistance is not 20')
    expected = 190.0 if perturbed else 20.0
    if max(abs(value - expected) for value in after) > tolerance:
        raise RuntimeError(
            'Post-ramp aortic resistance is not %g' % expected)
    return min(before + after), max(before + after)


def latest_completed_cycle_metrics(rows):
    cycles = []
    current = []
    for row in rows:
        if float(row.get('new_beat', 0.0)) > 0.0 and current:
            cycles.append(current)
            current = []
        current.append(row)
    if len(cycles) == 0:
        raise RuntimeError('CSV contains no completed cardiac cycle')
    cycle = cycles[-1]
    volumes = [float(row['volume_ventricle']) for row in cycle]
    pressures = [float(row['pressure_ventricle']) for row in cycle]
    edv = max(volumes)
    esv = min(volumes)
    stroke_volume = edv - esv
    return {'EDV': edv, 'ESV': esv, 'SV': stroke_volume,
            'EF_percent': 100.0*stroke_volume/edv,
            'peak_LV_pressure': max(pressures)}


def value(row, name):
    return float(row[name])


def validate_invariants(label, row, tolerance, myofiber_fraction):
    if abs(value(row, 'fib_reference_volume_residual')) > tolerance:
        raise RuntimeError('%s changed myocardial reference volume' % label)
    if abs(value(row, 'fib_total_mass_residual')) > tolerance:
        raise RuntimeError('%s changed total constituent mass' % label)
    for suffix in ('min', 'mean', 'max'):
        if abs(value(row, 'fib_phi_m_' + suffix) - myofiber_fraction) > tolerance:
            raise RuntimeError('%s changed myofiber fraction' % label)
    fraction_sum = (value(row, 'fib_phi_m_mean') +
                    value(row, 'fib_phi_g_mean') +
                    value(row, 'fib_phi_c_mean'))
    if abs(fraction_sum - 1.0) > tolerance:
        raise RuntimeError('%s mean fractions do not sum to one' % label)
    for name in ('fib_normalized_stimulus_min',
                 'fib_normalized_stimulus_mean',
                 'fib_normalized_stimulus_max',
                 'fib_effective_stimulus_min',
                 'fib_effective_stimulus_mean',
                 'fib_effective_stimulus_max', 'fib_gamma_c'):
        diagnostic = value(row, name)
        if math.isnan(diagnostic) or math.isinf(diagnostic):
            raise RuntimeError('%s has non-finite %s' % (label, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('healthy_csv')
    parser.add_argument('as_csv')
    parser.add_argument('--absolute-tolerance', type=float, default=1.0e-8)
    parser.add_argument('--healthy-collagen-tolerance', type=float,
                        default=5.0e-3)
    parser.add_argument('--initial-collagen-fraction', type=float,
                        default=0.03)
    parser.add_argument('--myofiber-fraction', type=float, default=0.70)
    args = parser.parse_args()
    healthy = final_row(args.healthy_csv)
    stenosis = final_row(args.as_csv)
    healthy_rows = rows_from(args.healthy_csv)
    stenosis_rows = rows_from(args.as_csv)
    validate_invariants('healthy', healthy, args.absolute_tolerance,
                        args.myofiber_fraction)
    validate_invariants('AS', stenosis, args.absolute_tolerance,
                        args.myofiber_fraction)
    healthy_phi = value(healthy, 'fib_phi_c_mean')
    as_phi = value(stenosis, 'fib_phi_c_mean')
    if abs(healthy_phi - args.initial_collagen_fraction) > \
            args.healthy_collagen_tolerance:
        raise RuntimeError('Healthy collagen accumulation is not negligible')
    if as_phi <= healthy_phi:
        raise RuntimeError('AS did not increase mean collagen over healthy')
    healthy_resistance = verify_aortic_resistance(
        healthy_rows, False, args.absolute_tolerance)
    stenosis_resistance = verify_aortic_resistance(
        stenosis_rows, True, args.absolute_tolerance)
    healthy_metrics = latest_completed_cycle_metrics(healthy_rows)
    stenosis_metrics = latest_completed_cycle_metrics(stenosis_rows)
    print('PASS: volume-neutral AS fibrosis invariants')
    print('healthy/AS final mean phi_c:', healthy_phi, as_phi)
    print('additional AS-associated mean phi_c:', as_phi - healthy_phi)
    print('healthy aortic-resistance range:', healthy_resistance)
    print('severe-AS aortic-resistance range:', stenosis_resistance)
    print('healthy final completed-cycle metrics:', healthy_metrics)
    print('severe-AS final completed-cycle metrics:', stenosis_metrics)
    print('gamma_c:', value(stenosis, 'fib_gamma_c'))
    print('AS raw stimulus min/mean/max:',
          value(stenosis, 'fib_normalized_stimulus_min'),
          value(stenosis, 'fib_normalized_stimulus_mean'),
          value(stenosis, 'fib_normalized_stimulus_max'))
    print('AS effective stimulus min/mean/max:',
          value(stenosis, 'fib_effective_stimulus_min'),
          value(stenosis, 'fib_effective_stimulus_mean'),
          value(stenosis, 'fib_effective_stimulus_max'))
    raw_range = observed_range(
        args.as_csv, 'fib_normalized_stimulus_min',
        'fib_normalized_stimulus_max')
    effective_range = observed_range(
        args.as_csv, 'fib_effective_stimulus_min',
        'fib_effective_stimulus_max')
    print('AS observed raw stimulus range over saved times:', raw_range)
    print('AS observed effective stimulus range over saved times:',
          effective_range)


if __name__ == '__main__':
    main()
