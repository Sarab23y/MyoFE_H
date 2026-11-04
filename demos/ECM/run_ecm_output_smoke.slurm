#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=ECM_output_smoke
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --partition=CAC48M192_L
#SBATCH --nodelist=cascadeb022
#SBATCH --account=col_jfwe223_uksr
#SBATCH --mail-user=Sara.b@uky.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_output_smoke.%J.out
#SBATCH --error=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_output_smoke.%J.err
set -euo pipefail

REPO=/home/sba431/MyoFE_H
IMAGE=/home/sba431/fenics.img
LEGACY_TEMPLATE=$REPO/demos/ECM/sim_input/base_instruction_ECM_output_smoke_legacy.json
BUFFERED_TEMPLATE=$REPO/demos/ECM/sim_input/base_instruction_ECM_output_smoke_buffered.json
ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM
RUN_ROOT=$ROOT/ECM_output_smoke_${SLURM_JOB_ID}
LEGACY=$RUN_ROOT/legacy
BUFFERED=$RUN_ROOT/buffered
LEGACY_JSON=$RUN_ROOT/config/legacy.json
BUFFERED_JSON=$RUN_ROOT/config/buffered.json
REPORT=$RUN_ROOT/comparison.json

for path in "$IMAGE" "$LEGACY_TEMPLATE" "$BUFFERED_TEMPLATE"; do
  [ -f "$path" ] || { echo "ERROR: missing $path" >&2; exit 2; }
done
[ ! -e "$RUN_ROOT" ] || {
  echo "ERROR: refusing to overwrite $RUN_ROOT" >&2; exit 3;
}
mkdir -p "$REPO/demos/ECM/logs" \
  "$RUN_ROOT/config" \
  "$LEGACY/sim_output" "$LEGACY/mesh_output" \
  "$BUFFERED/sim_output" "$BUFFERED/mesh_output"

singularity exec --cleanenv "$IMAGE" python - \
  "$LEGACY_TEMPLATE" "$BUFFERED_TEMPLATE" \
  "$LEGACY_JSON" "$BUFFERED_JSON" "$LEGACY" "$BUFFERED" <<'PY'
from __future__ import print_function
import json, sys
legacy, buffered = [json.load(open(path)) for path in sys.argv[1:3]]
output_jsons = sys.argv[3:5]
output_roots = sys.argv[5:7]
for case in (legacy, buffered):
    assert case['protocol']['time_step'][0] == 0.001
    assert case['protocol']['no_of_time_steps'][0] == 200
assert buffered['output_handler']['output_mode'][0] == 'buffered'
required_ecm = set(('ecm_instantaneous_fiber_stress',
                    'ecm_cycle_averaged_fiber_stress',
                    'ecm_homeostatic_fiber_stress'))
for case in (legacy, buffered):
    assert required_ecm.issubset(set(
        case['output_handler']['ecm_spatial_csv_fields']))
for case, path, root in zip((legacy, buffered), output_jsons, output_roots):
    case['output_handler']['output_data_path'] = [
        root + '/sim_output/data.csv']
    case['output_handler']['mesh_output_path'] = [root + '/mesh_output']
    with open(path, 'w') as handle:
        json.dump(case, handle, indent=2, sort_keys=True)
        handle.write('\n')
print('PASS: matched 200-step smoke configurations')
PY

cd "$REPO/python_codes"
singularity exec --cleanenv "$IMAGE" mpiexec -np "$SLURM_NTASKS" \
  python MyoFE.py LV_sim "$LEGACY_JSON"
singularity exec --cleanenv "$IMAGE" mpiexec -np "$SLURM_NTASKS" \
  python MyoFE.py LV_sim "$BUFFERED_JSON"

cd "$REPO"
singularity exec --cleanenv "$IMAGE" python \
  validation/compare_output_modes.py \
  "$LEGACY/sim_output" "$BUFFERED/sim_output" --report "$REPORT"
echo "PASS: legacy/buffered comparison: $REPORT"
