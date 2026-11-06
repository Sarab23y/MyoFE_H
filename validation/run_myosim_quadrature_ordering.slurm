#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --job-name=myosim_q_order
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L
#SBATCH --nodelist=cascadeb022
#SBATCH --output=/home/sba431/MyoFE_H/demos/ECM/logs/myosim_q_order.%J.out
#SBATCH --error=/home/sba431/MyoFE_H/demos/ECM/logs/myosim_q_order.%J.err
set -euo pipefail

REPO=${REPO:-/home/sba431/MyoFE_H}
IMAGE=${IMAGE:-/home/sba431/fenics.img}
INPUT=${INPUT:-$REPO/demos/ECM/sim_input/base_instruction_Br_ECM_40s_healthy_quad_validation_optimized.json}
OUT=${OUT:-/mnt/gpfs2_4m/scratch/sba431/demos/ECM/myosim_q_order_${SLURM_JOB_ID}}
mkdir -p "$OUT"

# Keep the established container-first MPI launch pattern.  No simulation is
# run: both commands only construct spaces and inspect their real dofmaps.
singularity exec --cleanenv "$IMAGE" \
    mpiexec -np 1 python \
    "$REPO/validation/validate_myosim_mixed_quadrature_ordering.py" \
    --input-json "$INPUT" --output "$OUT/one_rank.json"

singularity exec --cleanenv "$IMAGE" \
    mpiexec -np 32 python \
    "$REPO/validation/validate_myosim_mixed_quadrature_ordering.py" \
    --input-json "$INPUT" --output "$OUT/thirty_two_rank.json"

python - "$OUT/one_rank.json" "$OUT/thirty_two_rank.json" <<'PY'
from __future__ import print_function
import json, sys
reports = [json.load(open(path)) for path in sys.argv[1:]]
passed = all(report.get("passed") is True for report in reports)
print("Combined one/32-rank ordering result: %s" %
      ("PASS" if passed else "FAIL"))
sys.exit(0 if passed else 1)
PY
