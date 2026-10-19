#!/bin/bash
#SBATCH --time=12-00:00:00
#SBATCH --job-name=Br_signedrange_single
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --nodelist=skylake044
#SBATCH --output=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/signedrange_single.%J.out
#SBATCH --error=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/signedrange_single.%J.err
set -euo pipefail

REPO=/home/sba431/MyoFE_H
FENICS_IMAGE=/home/sba431/fenics.img
JSON="$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_signedrange_single.json"
RESULT_ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/signedrange_single

[ -f "$JSON" ] || { echo "ERROR: missing $JSON" >&2; exit 2; }
[ -f "$FENICS_IMAGE" ] || \
  { echo "ERROR: missing FEniCS image $FENICS_IMAGE" >&2; exit 2; }

# Fail before moving old results unless the one input has the exact requested
# protocol and automatic calibration mode.
singularity exec --cleanenv "$FENICS_IMAGE" python - "$JSON" <<'PY'
from __future__ import print_function
import json
import sys

path = sys.argv[1]
with open(path, 'r') as handle:
    case = json.load(handle)
first = lambda mapping, key: mapping[key][0]
protocol = case['protocol']
fibrosis = case['model']['fibrosis']
if first(protocol, 'time_step') != 0.001 or \
        first(protocol, 'no_of_time_steps') != 60000:
    raise RuntimeError('Expected 60 seconds at dt=0.001 seconds')
baro = protocol['baroreflex']['activations'][0]
window = protocol['growth']['activations'][0]
if (first(baro, 't_start_s'), first(baro, 't_stop_s')) != (5, 60):
    raise RuntimeError('Expected baroreflex activation from 5 to 60 seconds')
if (first(window, 't_start_s'), first(window, 't_stop_s')) != (20, 60):
    raise RuntimeError('Expected fibrosis activation from 20 to 60 seconds')
changes = protocol['perturbation']['perturbations']
if len(changes) != 1:
    raise RuntimeError('Expected exactly one perturbation')
change = changes[0]
actual = (first(change, 'level'), first(change, 'variable'),
          first(change, 't_start_s'), first(change, 't_stop_s'),
          first(change, 'total_change'))
if actual != ('circulation', 'aorta_resistance', 20, 30, 170):
    raise RuntimeError('Incorrect resistance perturbation: %r' % (actual,))
if 'growth' in case['model'] or \
        first(fibrosis, 'geometric_growth_enabled') is not False:
    raise RuntimeError('Geometric growth must remain disabled')
if first(fibrosis, 'stimulus') != \
        'cycle_averaged_total_fiber_stress_range':
    raise RuntimeError('Signed-range stimulus is not selected')
calibration = fibrosis['stress_reference_calibration']
expected = (True, 'global_spatial_percentile', 3, 1.0, 99.0)
actual = (first(calibration, 'enabled'), first(calibration, 'method'),
          first(calibration, 'cycles'),
          first(calibration, 'lower_percentile'),
          first(calibration, 'upper_percentile'))
if actual != expected:
    raise RuntimeError('Incorrect automatic calibration: %r' % (actual,))
if first(fibrosis, 'gamma_c') != 1.5 or first(fibrosis, 'tau_c') != 50:
    raise RuntimeError('Incorrect gamma_c or tau_c')
print('PASS: one 60-second baseline-to-overload signed-range input')
print('PASS: automatic references use the last 3 pre-20-s cycles')
print('PASS: aorta_resistance ramps by +170 from 20 to 30 s')
PY

STAMP=$(date +%Y%m%d-%H%M%S)
if [ -d "$RESULT_ROOT" ]; then
  mv "$RESULT_ROOT" "${RESULT_ROOT}.backup.${STAMP}"
fi
mkdir -p "$RESULT_ROOT/sim_output" "$RESULT_ROOT/mesh_output"

cd "$REPO/python_codes"
if ! singularity exec --cleanenv "$FENICS_IMAGE" \
    mpiexec -np "$SLURM_NTASKS" python MyoFE.py LV_sim "$JSON"; then
  echo "ERROR: signed-range LV solver failed" >&2
  exit 1
fi
echo "PASS: signed-range LV simulation completed: $RESULT_ROOT"
