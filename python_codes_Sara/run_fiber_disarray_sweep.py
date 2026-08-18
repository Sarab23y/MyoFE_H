#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import subprocess


def build_case(base_data, theta_deg, ell_c, seed, out_dir):
    data = copy.deepcopy(base_data)
    mesh = data['mesh']
    mesh['fiber_architecture'] = ['disarray']
    mesh['theta_rms_deg'] = [float(theta_deg)]
    mesh['ell_c'] = [None if ell_c is None else float(ell_c)]
    mesh['disarray_seed'] = [int(seed)]

    case_tag = 'theta_%s_ell_%s_seed_%d' % (
        str(theta_deg).replace('.', 'p'),
        'none' if ell_c is None else str(ell_c).replace('.', 'p'),
        int(seed)
    )

    if 'output_handler' in data:
        root = os.path.join(out_dir, case_tag)
        data['output_handler']['output_data_path'] = [os.path.join(root, 'data.csv')]
        data['output_handler']['mesh_output_path'] = [os.path.join(root, 'mesh_output')]

    return data, case_tag


def main():
    ap = argparse.ArgumentParser(description='Run/generate fiber disarray sweep inputs')
    ap.add_argument('--base-json', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--thetas', default='15,20,25')
    ap.add_argument('--ell-c', default='0.075')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--mpiexec', default='mpiexec')
    ap.add_argument('--np', type=int, default=1)
    args = ap.parse_args()

    with open(args.base_json, 'r') as f:
        base_data = json.load(f)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    thetas = [float(x) for x in args.thetas.split(',') if x.strip()]
    if args.ell_c.strip().lower() == 'none':
        ell_values = [None]
    else:
        ell_values = [float(x) for x in args.ell_c.split(',') if x.strip()]

    summary = []
    for theta in thetas:
        for ell_c in ell_values:
            case_data, case_tag = build_case(base_data, theta, ell_c, args.seed, args.out_dir)
            case_json = os.path.join(args.out_dir, case_tag + '.json')
            with open(case_json, 'w') as f:
                json.dump(case_data, f, indent=4)

            summary.append({'case': case_tag, 'json': case_json, 'theta_rms_deg': theta, 'ell_c': ell_c, 'seed': args.seed})

            if args.run:
                cmd = [args.mpiexec, '-np', str(args.np), 'python', 'MyoFE.py', 'LV_sim', case_json]
                subprocess.check_call(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(args.out_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print('Generated %d sweep case(s)' % len(summary))


if __name__ == '__main__':
    main()
