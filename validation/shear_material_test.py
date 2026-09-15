#!/usr/bin/env python
"""Run isochoric local-coordinate shear tests of the production passive law."""
from __future__ import print_function

import argparse
import os

import numpy as np

from material_point import (evaluate_state, load_material_input, shear_F,
                            write_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    config, input_path, passive, resolved = load_material_input(args.config)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(args.config)),
                              config['output_directory'])
    shear_values = np.linspace(config['shear_min'], config['shear_max'],
                               int(config['number_of_points']))
    for direction in config['shear_directions']:
        rows = []
        for gamma in shear_values:
            row = evaluate_state(shear_F(float(gamma), direction), passive)
            row['direction'] = direction
            row['prescribed_gamma'] = float(gamma)
            rows.append(row)
        columns = ['direction', 'prescribed_gamma'] + sorted(
            key for key in rows[0]
            if key not in ('direction', 'prescribed_gamma'))
        csv_path = os.path.join(output_dir, 'shear_%s.csv' % direction)
        metadata = {
            'test': 'prescribed isochoric simple shear',
            'direction': direction,
            'input_json': input_path,
            'passive_parameters': resolved,
            'constraint': 'J=1; one total pressure from sigma_nn=0',
            'note': ('No explicit fiber-sheet coupling energy exists. Shear '
                     'can activate I1 and a stretched directional I4.'),
        }
        metadata_path = write_results(rows, columns, csv_path, metadata)
        print('Wrote %s and %s' % (csv_path, metadata_path))


if __name__ == '__main__':
    main()
