#!/bin/bash
#SBATCH --job-name=FEcycle_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --account=col_jfwe223_uksr
#SBATCH --output=FEcycle_smoke_%j.out
#SBATCH --error=FEcycle_smoke_%j.err
# Partition/node must be selected after checking availability; never pin cascadeb024.
set -euo pipefail
: "${REPO:?Set isolated diagnostic worktree}"
: "${CASE:?Set prepared instruction.json}"
: "${ANALYSIS_PYTHON:?Set Python3.7+ with NumPy h5py and Matplotlib}"
IMAGE=${IMAGE:-/home/sba431/fenics.img}
FENICS_PYTHON=${FENICS_PYTHON:-python}
test -f "$CASE"
test -f "$IMAGE"
test -f "$REPO/validation/run_fe_cycle_case.py"
"$ANALYSIS_PYTHON" -c 'import numpy,h5py,matplotlib'
cd "$REPO/python_codes"
NEEDS_NATIVE=$(singularity exec --cleanenv "$IMAGE" "$FENICS_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1]))["fe_cycle_diagnostic"]; print(int(d["enabled"] or d.get("invariance_witness",False)))' "$CASE")
STATUS=0
if test "$NEEDS_NATIVE" = 1; then
  : "${NATIVE_HOOK:?Set library compiled against this exact container}"
  test -f "$NATIVE_HOOK"
  singularity exec --cleanenv "$IMAGE" mpiexec -np "${SLURM_NTASKS:-32}" \
    env LD_PRELOAD="$NATIVE_HOOK" MYOFE_NEWTON_OBSERVER="$NATIVE_HOOK" \
    "$FENICS_PYTHON" "$REPO/validation/run_fe_cycle_case.py" "$CASE" || STATUS=$?
else
  singularity exec --cleanenv "$IMAGE" mpiexec -np "${SLURM_NTASKS:-32}" \
    "$FENICS_PYTHON" "$REPO/validation/run_fe_cycle_case.py" "$CASE" || STATUS=$?
fi
OBSERVER=$(singularity exec --cleanenv "$IMAGE" "$FENICS_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["fe_cycle_diagnostic"]["directory"])' "$CASE")
if test "${EXPECT_INTERRUPTED:-0}" = 1; then
  test "$STATUS" != 0
  "$ANALYSIS_PYTHON" "$REPO/validation/check_fe_cycle_output.py" \
    --instruction "$CASE" --expect-interrupted --output "$(dirname "$OBSERVER")/interruption_integrity.json"
  exit 0
fi
if test "$STATUS" != 0; then exit "$STATUS"; fi
if test -d "$OBSERVER"; then
  singularity exec --cleanenv "$IMAGE" mpiexec -np 1 "$FENICS_PYTHON" \
    "$REPO/validation/validate_fe_cycle_checkpoint.py" \
    --state "$OBSERVER/frames/state_00000000.h5" --output "$OBSERVER/reopen_1.json"
  if test "${SLURM_NTASKS:-32}" = 32; then
    singularity exec --cleanenv "$IMAGE" mpiexec -np 32 "$FENICS_PYTHON" \
      "$REPO/validation/validate_fe_cycle_checkpoint.py" \
      --state "$OBSERVER/frames/state_00000000.h5" --output "$OBSERVER/reopen_32.json"
  fi
fi
# Use a separately configured NumPy/h5py Python 3 for reporting outside the image.
if test -n "${ANALYSIS_PYTHON:-}" && test -d "$OBSERVER"; then
  "$ANALYSIS_PYTHON" "$REPO/postprocessing/report_fe_cycle_case.py" \
    --instruction "$CASE"
fi
"$ANALYSIS_PYTHON" "$REPO/validation/check_fe_cycle_output.py" \
  --instruction "$CASE" --output "$(dirname "$OBSERVER")/output_integrity.json"
