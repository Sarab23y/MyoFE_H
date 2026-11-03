#!/usr/bin/env python
"""Summarize measured fiber-stress samples without selecting model bounds."""
from __future__ import print_function

import argparse
import csv
import math


def percentile(values, percentage):
    position = (len(values) - 1)*percentage/100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return (values[lower]*(upper-position) + values[upper]*(position-lower))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples', help='CSV of stress samples')
    args = parser.parse_args()
    values = []
    with open(args.samples, 'r') as handle:
        for row in csv.reader(handle):
            for item in row:
                try:
                    value = float(item)
                except ValueError:
                    continue
                if not math.isnan(value) and not math.isinf(value):
                    values.append(value)
    values.sort()
    if not values:
        raise RuntimeError('No finite stress samples were found')
    print('finite sample count:', len(values))
    for percentage in (0, 0.1, 1, 5, 50, 95, 99, 99.9, 100):
        print('percentile %5.1f: %.17g' %
              (percentage, percentile(values, percentage)))
    print('This report does not choose S_ff_min or S_ff_max.')


if __name__ == '__main__':
    main()
