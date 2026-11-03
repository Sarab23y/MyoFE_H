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
LEGACY_JSON=$REPO/demos/ECM/sim_input/base_instruction_ECM_output_smoke_legacy.json
BUFFERED_JSON=$REPO/demos/ECM/sim_input/base_instruction_ECM_output_smoke_buffered.json
ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM
LEGACY=$ROOT/ECM_output_smoke_legacy
BUFFERED=$ROOT/ECM_output_smoke_buffered
REPORT=$ROOT/ECM_output_smoke_comparison.json

for path in "$IMAGE" "$LEGACY_JSON" "$BUFFERED_JSON"; do
  [ -f "$path" ] || { echo "ERROR: missing $path" >&2; exit 2; }
done
for result in "$LEGACY" "$BUFFERED" "$REPORT"; do
  [ ! -e "$result" ] || { echo "ERROR: refusing to overwrite $result" >&2; exit 3; }
done
mkdir -p "$REPO/demos/ECM/logs" \
  "$LEGACY/sim_output" "$LEGACY/mesh_output" \
  "$BUFFERED/sim_output" "$BUFFERED/mesh_output"

singularity exec --cleanenv "$IMAGE" python - "$LEGACY_JSON" "$BUFFERED_JSON" <<'PY'
from __future__ import print_function
import json, sys
legacy, buffered = [json.load(open(path)) for path in sys.argv[1:]]
for case in (legacy, buffered):
    assert case['protocol']['time_step'][0] == 0.001
    assert case['protocol']['no_of_time_steps'][0] == 200
assert buffered['output_handler']['output_mode'][0] == 'buffered'
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
