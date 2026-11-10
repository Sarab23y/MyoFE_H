#!/bin/bash
#SBATCH --job-name=phase_region
#SBATCH --output=phase_region_%j.out
#SBATCH --error=phase_region_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L
set -euo pipefail
REPO=${REPO:-/home/sba431/MyoFE_H_diagnostics}
DATA=${DATA:-/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/sim_output}
INPUT=${INPUT:-$(dirname "$DATA")/instruction.json}
: "${MAPPING:?Set MAPPING to the completed mapping job report directory}"
: "${LONG_AXIS:?Set LONG_AXIS to 0, 1, or 2 after anatomical review}"
: "${ANALYSIS_PYTHON:?Set ANALYSIS_PYTHON to your separate Python 3 executable with NumPy}"
: "${WALL_DEPTH_VERIFIED:?Set WALL_DEPTH_VERIFIED=1 after anatomical wall-depth review}"
test "$WALL_DEPTH_VERIFIED" = 1
REPO=$(realpath "$REPO")
PRODUCTION_REPO=$(realpath -m /home/sba431/MyoFE_H)
if test "$REPO" = "$PRODUCTION_REPO"; then
  echo 'Refusing to run from the production checkout.' >&2
  exit 2
fi
test "$(git -C "$REPO" rev-parse --show-toplevel)" = "$REPO"
test -f "$INPUT"
OUT=${OUT:-/mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_region_${SLURM_JOB_ID}}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$ANALYSIS_PYTHON" -c 'import sys,numpy; assert sys.version_info >= (3,6); print(sys.version); print(numpy.__version__)'
cd "$REPO"
WINDOW=()
if test -n "${START:-}"; then WINDOW+=(--start "$START"); fi
if test -n "${END:-}"; then WINDOW+=(--end "$END"); fi
# The analyzer itself verifies the sealed suite, input/source/report hashes,
# geometry and weights. Direct CLI use cannot bypass the suite gate.
"$ANALYSIS_PYTHON" validation/analyze_phase_region.py \
  --data "$DATA" --mapping "$MAPPING" --input-json "$INPUT" \
  --axis "$LONG_AXIS" "${WINDOW[@]}" --cycles 5 --output "$OUT"
echo "Reports: $OUT"
