#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python2 python_codes_org/MyoFE.py LV_sim demos/finer_mesh/baseline_15_cycles_no_baroreflex_no_disarray.json
