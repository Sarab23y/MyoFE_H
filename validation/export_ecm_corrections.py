#!/usr/bin/env python3
"""Build a small, auditable transfer bundle for the ECM diagnostic fixes.

The bundle is deliberately allow-listed.  It cannot pick up simulation output,
mesh data, Git metadata, or unrelated working-tree files.
"""
from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import zipfile


FILES = (
    'python_codes/LV_simulation/fibrosis/__init__.py',
    'python_codes/LV_simulation/fibrosis/ecm_stress.py',
    'python_codes/LV_simulation/dependencies/ecm_stability.py',
    'python_codes/LV_simulation/dependencies/ecm_stress_diagnostic.py',
    'python_codes/LV_simulation/LV_simulation.py',
    'python_codes/MyoFE.py',
    'validation/analyze_ecm_spatial_anomalies.py',
    'tests/test_ecm_spatial_analysis.py',
    'tests/test_ecm_stress_diagnostic.py',
    'docs/ECM_Spatial_Stress_Investigation.md',
)


def git(root, *args):
    return subprocess.check_output(
        ['git'] + list(args), cwd=root, stderr=subprocess.STDOUT)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def instructions(baseline, head):
    return """# Applying the ECM correction transfer bundle

This archive contains an allow-listed overlay plus a cumulative Git patch.

Patch baseline: `{baseline}`
Exported head: `{head}`

The user's GitHub master (`8c1f57d4a6dee30b962460eb4759e54dfbcf3a49`)
was not available in the isolated export checkout.  Therefore, do not force the
patch if `git apply --check` reports conflicts.

## Safe application procedure

```bash
git fetch origin --prune
git switch -c review/ecm-spatial-corrections \
  8c1f57d4a6dee30b962460eb4759e54dfbcf3a49

unzip MyoFE_H_ECM_corrections_{short}.zip -d /tmp/ecm-transfer
cd /path/to/MyoFE_H

# Inspect the manifest and patch before changing the checkout.
cat /tmp/ecm-transfer/TRANSFER_MANIFEST.json
git apply --stat /tmp/ecm-transfer/ecm_corrections.patch
git apply --check /tmp/ecm-transfer/ecm_corrections.patch
```

If the check succeeds:

```bash
git apply --3way /tmp/ecm-transfer/ecm_corrections.patch
git status --short
git diff --check
git diff --stat
git diff
python -m unittest discover -s tests -v
```

If the check fails, do **not** use `--reject` or copy the complete driver
blindly.  Compare each allow-listed file against `files/` and port only the
relevant hunks, for example:

```bash
git diff --no-index -- \
  python_codes/LV_simulation/fibrosis/ecm_stress.py \
  /tmp/ecm-transfer/files/python_codes/LV_simulation/fibrosis/ecm_stress.py
```

After resolving and testing, commit on the review branch and open a pull
request.  Do not merge until the FEniCS 2019.1 one-rank/32-rank quadrature
runtime validation described in `docs/ECM_Spatial_Stress_Investigation.md` has
been reviewed.
""".format(baseline=baseline, head=head, short=head[:7])


def build(root, output, baseline, head):
    root = os.path.abspath(root)
    head = git(root, 'rev-parse', head).decode('ascii').strip()
    baseline = git(root, 'rev-parse', baseline).decode('ascii').strip()
    entries = []
    payloads = {}
    for path in FILES:
        absolute = os.path.join(root, path)
        if not os.path.isfile(absolute):
            raise RuntimeError('Required transfer file is missing: %s' % path)
        with open(absolute, 'rb') as handle:
            data = handle.read()
        payloads[path] = data
        entries.append({'path': path, 'bytes': len(data),
                        'sha256': sha256(data)})

    patch = git(root, 'diff', '--binary', baseline + '..' + head, '--', *FILES)
    manifest = {
        'format_version': 1,
        'created_utc': datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(),
        'patch_baseline': baseline,
        'exported_head': head,
        'github_master_reported_by_user':
            '8c1f57d4a6dee30b962460eb4759e54dfbcf3a49',
        'github_master_available_in_export_checkout': False,
        'files': entries,
        'patch': {'path': 'ecm_corrections.patch', 'bytes': len(patch),
                  'sha256': sha256(patch)},
        'exclusions': ['raw simulation output', 'mesh/HDF5/XDMF data',
                       'images', 'Git metadata', 'unrelated source files'],
    }
    manifest_data = json.dumps(
        manifest, indent=2, sort_keys=True).encode('utf-8') + b'\n'
    readme = instructions(baseline, head).encode('utf-8')

    output = os.path.abspath(output)
    parent = os.path.dirname(output)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('TRANSFER_MANIFEST.json', manifest_data)
        archive.writestr('APPLY_INSTRUCTIONS.md', readme)
        archive.writestr('ecm_corrections.patch', patch)
        for path, data in payloads.items():
            archive.writestr('files/' + path, data)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--baseline', default='daab60c')
    parser.add_argument('--head', default='HEAD')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    manifest = build(args.repo, args.output, args.baseline, args.head)
    print('Wrote:', os.path.abspath(args.output))
    print('Files:', len(manifest['files']))
    print('Patch baseline:', manifest['patch_baseline'])
    print('Exported head:', manifest['exported_head'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
