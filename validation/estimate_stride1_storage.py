#!/usr/bin/env python
"""Conservative storage estimate for the 60-second all-output profile."""
from __future__ import print_function

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path')
    parser.add_argument('--quadrature-points', type=int, default=5008)
    parser.add_argument('--mesh-points', type=int, default=442)
    parser.add_argument('--spatial-fields', type=int, default=70,
                        help='conservative all-output scalar field count')
    parser.add_argument('--global-columns', type=int, default=21000,
                        help='includes per-point baroreflex control signals')
    parser.add_argument('--bytes-per-csv-value', type=int, default=24)
    args = parser.parse_args()
    config = json.load(open(args.json_path))
    steps = int(config['protocol']['no_of_time_steps'][0])
    rows = steps + 1
    output = config['output_handler']
    mesh_fields = len(output.get('mesh_object_to_save', []))
    spatial = rows * args.spatial_fields * \
        (args.quadrature_points + 2) * args.bytes_per_csv_value
    # HDF5 is binary; use 32 bytes/value to cover datasets and XDMF metadata.
    mesh = rows * mesh_fields * args.mesh_points * 32
    global_csv = rows * args.global_columns * args.bytes_per_csv_value
    raw = spatial + mesh + global_csv
    required = int(raw * 1.5)  # filesystem/metadata and field-count margin
    report = {'rows': rows, 'quadrature_points': args.quadrature_points,
              'assumed_spatial_fields': args.spatial_fields,
              'assumed_global_columns': args.global_columns,
              'mesh_fields': mesh_fields, 'estimated_raw_bytes': raw,
              'required_free_bytes_with_50_percent_margin': required,
              'required_free_gib': required / float(1024 ** 3)}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
