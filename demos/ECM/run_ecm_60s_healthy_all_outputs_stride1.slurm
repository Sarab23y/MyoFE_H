#!/bin/bash
#SBATCH --job-name=ecm60_all_stride1
#SBATCH --output=ecm60_all_stride1_%j.out
#SBATCH --error=ecm60_all_stride1_%j.err
#SBATCH --time=12-00:00:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --partition=CAC48M192_L
#SBATCH --nodelist=cascadeb024
#SBATCH --account=col_jfwe223_uksr

set -euo pipefail
IMAGE=/home/sba431/fenics.img
REPO=/home/sba431/MyoFE_H
JSON="$REPO/demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json"
OUT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1
SMOKE_GLOB=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1_smoke_\*/sim_output/stride1_validation.json

test -f "$IMAGE"
test -f "$JSON"
SMOKE_REPORT=$(ls -1t $SMOKE_GLOB 2>/dev/null | head -1 || true)
if test -z "$SMOKE_REPORT" || ! python - "$SMOKE_REPORT" <<'PY'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get('passed') else 1)
PY
then
    echo "No passing stride-one smoke report found; refusing full run." >&2
    exit 4
fi
echo "Using passing smoke report: $SMOKE_REPORT"
if test -e "$OUT"; then
    echo "Refusing to overwrite existing output: $OUT" >&2
    exit 2
fi

ESTIMATE=$(python "$REPO/validation/estimate_stride1_storage.py" "$JSON")
echo "$ESTIMATE"
REQUIRED=$(python -c 'import json,sys; print(json.load(sys.stdin)["required_free_bytes_with_50_percent_margin"])' <<<"$ESTIMATE")
PARENT=$(dirname "$OUT")
AVAILABLE=$(df -Pk "$PARENT" | awk 'NR==2 {print $4 * 1024}')
echo "available_bytes=$AVAILABLE required_bytes=$REQUIRED"
if python - "$AVAILABLE" "$REQUIRED" <<'PY'
import sys
sys.exit(0 if int(float(sys.argv[1])) >= int(sys.argv[2]) else 1)
PY
then :; else
    echo "Insufficient free space; output frequency/fields were not reduced." >&2
    exit 3
fi

mkdir -p "$OUT"
printf '%s\n' "$ESTIMATE" > "$OUT/storage_estimate.json"
cp "$JSON" "$OUT/instruction.json"

cd "$REPO/python_codes"
singularity exec --cleanenv "$IMAGE" \
    mpiexec -np "$SLURM_NTASKS" \
    python MyoFE.py LV_sim "$JSON"

singularity exec --cleanenv "$IMAGE" python \
    "$REPO/validation/validate_all_outputs_stride1.py" \
    "$OUT/sim_output" --expected-rows 60001 --dt 0.001
echo "PASS: $OUT"
