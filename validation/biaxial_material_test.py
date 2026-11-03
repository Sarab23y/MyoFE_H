#!/usr/bin/env python
"""Run prescribed-stretch biaxial tests of the production passive law."""
from __future__ import print_function

import argparse
import json
import os

import numpy as np

from material_point import (biaxial_F, evaluate_state, load_material_input,
                            write_results)


PATHS = {
    'equal_biaxial': lambda value: (value, value),
    'fiber_dominant': lambda value: (value, 1.0 + 0.5*(value - 1.0)),
    'sheet_dominant': lambda value: (1.0 + 0.5*(value - 1.0), value),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    config, input_path, passive, resolved = load_material_input(args.config)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(args.config)),
                              config['output_directory'])
    stretches = np.linspace(config['stretch_min'], config['stretch_max'],
                            int(config['number_of_points']))
    for path_name in config['biaxial_paths']:
        if path_name not in PATHS:
            raise ValueError('Unknown biaxial path: %s' % path_name)
        rows = []
        for value in stretches:
            lambda_f, lambda_s = PATHS[path_name](float(value))
            row = evaluate_state(biaxial_F(lambda_f, lambda_s), passive)
            row['path'] = path_name
            rows.append(row)
        columns = ['path'] + sorted(key for key in rows[0] if key != 'path')
        csv_path = os.path.join(output_dir, 'biaxial_%s.csv' % path_name)
        metadata = {
            'test': 'prescribed-stretch incompressible biaxial test',
            'path': path_name,
            'path_definition': {
                'equal_biaxial': 'lambda_f=lambda_s=value',
                'fiber_dominant': 'lambda_f=value; lambda_s=1+0.5(value-1)',
                'sheet_dominant': 'lambda_s=value; lambda_f=1+0.5(value-1)',
            }[path_name],
            'input_json': input_path,
            'passive_parameters': resolved,
            'constraint': ('J=1; one total pressure sets normal traction to '
                           'zero on the deformed thickness face'),
            'surface_traction': ('Aligned diagonal biaxial kinematics have '
                                 'zero tangential traction as well'),
            'stress_units': 'production internal stress unit (Pa for current input)',
        }
        metadata_path = write_results(rows, columns, csv_path, metadata)
        print('Wrote %s and %s' % (csv_path, metadata_path))


if __name__ == '__main__':
    main()
