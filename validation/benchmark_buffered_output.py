#!/usr/bin/env python
"""Dependency-free synthetic memory/timing benchmark for output buffering."""
from __future__ import division, print_function
import argparse
import json
import resource
import time
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from python_codes.LV_simulation.dependencies.buffered_output import \
    estimated_numeric_buffer_bytes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, default=2500)
    parser.add_argument('--fields', type=int, default=13)
    parser.add_argument('--saved-rows', type=int, default=4201)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--output', default='buffered_output_benchmark.json')
    args = parser.parse_args(argv)
    start = time.time()
    legacy = estimated_numeric_buffer_bytes(
        args.fields, args.points, args.saved_rows)
    buffered = estimated_numeric_buffer_bytes(
        args.fields, args.points, args.batch_size)
    report = {'kind': 'analytical_float64_payload_and_process_measurement',
              'points': args.points, 'fields': args.fields,
              'saved_rows': args.saved_rows, 'batch_size': args.batch_size,
              'legacy_numeric_payload_bytes': legacy,
              'buffered_numeric_payload_bytes': buffered,
              'payload_reduction_factor': legacy/buffered,
              'benchmark_wall_seconds': time.time()-start,
              'process_max_rss_kib': resource.getrusage(
                  resource.RUSAGE_SELF).ru_maxrss,
              'limitation': 'Payload estimate excludes pandas/Python overhead; '
                            'production MPI/FEniCS timings require the cluster run.'}
    with open(args.output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
