#!/bin/bash
#SBATCH --job-name=ecm60_all_smoke
#SBATCH --output=ecm60_all_smoke_%j.out
#SBATCH --error=ecm60_all_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --partition=CAC48M192_L
#SBATCH --nodelist=cascadeb022
#SBATCH --account=col_jfwe223_uksr

set -euo pipefail
IMAGE=/home/sba431/fenics.img
REPO=/home/sba431/MyoFE_H
TEMPLATE="$REPO/demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1_smoke.json"
OUT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1_smoke_${SLURM_JOB_ID}
RUNTIME_JSON="$OUT/instruction.json"

test -f "$IMAGE"
test -f "$TEMPLATE"
test ! -e "$OUT"
mkdir -p "$OUT"
python - "$TEMPLATE" "$RUNTIME_JSON" "$OUT" <<'PY'
import json, sys
source, target, output = sys.argv[1:]
data = json.load(open(source))
handler = data['output_handler']
handler['output_data_path'] = [output + '/sim_output/data.csv']
handler['mesh_output_path'] = [output + '/mesh_output']
with open(target, 'w') as stream:
    json.dump(data, stream, indent=2)
PY

cd "$REPO/python_codes"
singularity exec --cleanenv "$IMAGE" \
    mpiexec -np "$SLURM_NTASKS" \
    python MyoFE.py LV_sim "$RUNTIME_JSON"

singularity exec --cleanenv "$IMAGE" python \
    "$REPO/validation/validate_all_outputs_stride1.py" \
    "$OUT/sim_output" --expected-rows 201 --dt 0.001
echo "PASS: $OUT"
