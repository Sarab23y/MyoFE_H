#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate baseline and/or disarray case JSON files from ONE master template JSON.
The input JSON is treated as source of truth: all fields are preserved unless
explicitly overridden for case generation.
"""

import argparse
import copy
import json
import os
import re


def _get_passive_law(data):
    return data['mesh']['forms_parameters']['passive_law_parameters']['passive_law'][0]


def _family_tag_from_law(law):
    return 'HO' if str(law).lower() == 'holzapfel' else 'Guccione'


def _ensure_output_handler(data):
    if 'output_handler' not in data:
        data['output_handler'] = {}
    oh = data['output_handler']
    if 'output_data_path' not in oh:
        oh['output_data_path'] = ['data_disarray.csv']
    if 'mesh_output_path' not in oh:
        oh['mesh_output_path'] = ['mesh_output_disarray']
    if 'dumping_spatial_in_average' not in oh:
        oh['dumping_spatial_in_average'] = [True]
    if 'frequency_n' not in oh:
        oh['frequency_n'] = [10]
    # Keep CSV-only outputs enabled by default.
    if 'save_outputs' not in oh:
        oh['save_outputs'] = ['data.csv', 'spatial_average.csv']


def _derive_case_output_paths(data, case_tag):
    oh = data['output_handler']
    data_path = oh['output_data_path'][0]
    mesh_path = oh['mesh_output_path'][0]

    data_name = os.path.basename(data_path) if data_path else 'data_disarray.csv'
    mesh_name = os.path.basename(mesh_path) if mesh_path else 'mesh_output_disarray'

    data_dir = os.path.dirname(data_path)
    mesh_dir = os.path.dirname(mesh_path)

    # Replace trailing /thetaXX with case tag when present, else append case tag.
    if re.search(r'/theta\d+/?$', data_dir):
        root = re.sub(r'/theta\d+/?$', '', data_dir)
        new_data_dir = os.path.join(root, case_tag)
    else:
        new_data_dir = os.path.join(data_dir, case_tag) if data_dir else case_tag

    if re.search(r'/theta\d+/?$', mesh_dir):
        root = re.sub(r'/theta\d+/?$', '', mesh_dir)
        new_mesh_dir = os.path.join(root, case_tag)
    else:
        new_mesh_dir = os.path.join(mesh_dir, case_tag) if mesh_dir else case_tag

    oh['output_data_path'] = [os.path.join(new_data_dir, data_name)]
    oh['mesh_output_path'] = [os.path.join(new_mesh_dir, mesh_name)]

    # Explicit CSV/spatial-average paths for consistency.
    oh['spatial_average_output_path'] = [os.path.join(new_data_dir, 'spatial_average.csv')]


def _apply_baseline_settings(data):
    mesh = data['mesh']
    mesh['fiber_architecture'] = ['aligned']
    if 'theta_rms_deg' in mesh:
        mesh['theta_rms_deg'] = [0.0]
    if 'disarray_width' in mesh:
        mesh['disarray_width'] = [0.0]


def _apply_disarray_settings(data, theta, seed, ell_c, mask):
    mesh = data['mesh']
    mesh['fiber_architecture'] = ['disarray']
    mesh['theta_rms_deg'] = [float(theta)]
    mesh['disarray_seed'] = [int(seed)]
    mesh['ell_c'] = [float(ell_c)]
    mesh['disarray_region_mask'] = [mask]


def generate_cases(input_json, out_dir, baseline=False, thetas=None, seed=42,
                   ell_c=0.075, mask='lv_midwall_no_apex_no_base'):
    with open(input_json, 'r') as f:
        base = json.load(f)

    law = _get_passive_law(base)
    family = _family_tag_from_law(law)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    generated = []

    if baseline:
        d = copy.deepcopy(base)
        _ensure_output_handler(d)
        _apply_baseline_settings(d)
        _derive_case_output_paths(d, 'baseline')
        out_name = 'Baseline_%s.json' % family
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, 'w') as f:
            json.dump(d, f, indent=4)
        generated.append(out_path)

    for theta in (thetas or []):
        d = copy.deepcopy(base)
        _ensure_output_handler(d)
        _apply_disarray_settings(d, theta=theta, seed=seed, ell_c=ell_c, mask=mask)
        _derive_case_output_paths(d, 'theta%d' % int(theta))
        out_name = 'Fiber_disarray_%s_theta%d.json' % (family, int(theta))
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, 'w') as f:
            json.dump(d, f, indent=4)
        generated.append(out_path)

    return generated


def main():
    ap = argparse.ArgumentParser(description='Generate baseline/disarray JSONs from one template JSON')
    ap.add_argument('--input-json', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--baseline', action='store_true', help='Generate baseline no-disarray case')
    ap.add_argument('--thetas', default='', help='Comma-separated theta values, e.g. 10,20,30')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ell-c', type=float, default=0.075)
    ap.add_argument('--mask', default='lv_midwall_no_apex_no_base')
    args = ap.parse_args()

    thetas = [int(float(x)) for x in args.thetas.split(',') if x.strip()]
    generated = generate_cases(
        input_json=args.input_json,
        out_dir=args.out_dir,
        baseline=args.baseline,
        thetas=thetas,
        seed=args.seed,
        ell_c=args.ell_c,
        mask=args.mask
    )
    print('Generated %d file(s)' % len(generated))
    for p in generated:
        print(p)


if __name__ == '__main__':
    main()
