#!/bin/bash
#SBATCH --job-name=FEcycle_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=col_jfwe223_uksr
#SBATCH --output=FEcycle_preflight_%j.out
#SBATCH --error=FEcycle_preflight_%j.err
set -euo pipefail
: "${REPO:?Set isolated worktree}"
: "${INPUT:?Set frozen prepared input with absolute production mesh path}"
: "${OUT:?Set new preflight output directory}"
IMAGE=${IMAGE:-/home/sba431/fenics.img}
FENICS_PYTHON=${FENICS_PYTHON:-python}
test "${SLURM_NTASKS:-0}" = 32
test -f "$INPUT"
test -f "$IMAGE"
test ! -e "$OUT"
mkdir "$OUT"
trap 'status=$?; printf "exit=%s line=%s\n" "$status" "$LINENO" > "$OUT/failure.txt"; exit "$status"' ERR
cp "$INPUT" "$OUT/instruction.json"
sha256sum "$IMAGE" "$INPUT" > "$OUT/input_image.sha256"
git -C "$REPO" status --short > "$OUT/worktree_status.txt"
cd "$REPO/python_codes"
singularity exec --cleanenv "$IMAGE" "$FENICS_PYTHON" -c \
  'import sys; sys.path.insert(0,sys.argv[1]); import MyoFE; import dolfin; print(sys.version); print(dolfin.__version__)' \
  "$REPO/python_codes" > "$OUT/model_import.log" 2>&1
singularity exec --cleanenv "$IMAGE" "$FENICS_PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert not d["mesh"]["relative_path"][0], "Prepare absolute mesh path before preflight"' "$INPUT"
singularity exec --cleanenv "$IMAGE" "$FENICS_PYTHON" \
  "$REPO/validation/build_native_newton_observer.py" --output "$OUT/native"
NATIVE_HOOK="$OUT/native/libmyofe_newton_observer.so"
for N in 1 32; do
  singularity exec --cleanenv "$IMAGE" mpiexec -np "$N" \
    env LD_PRELOAD="$NATIVE_HOOK" MYOFE_NEWTON_OBSERVER="$NATIVE_HOOK" \
    "$FENICS_PYTHON" "$REPO/validation/smoke_native_newton.py" --output "$OUT/native_$N"
  singularity exec --cleanenv "$IMAGE" mpiexec -np "$N" "$FENICS_PYTHON" \
    "$REPO/validation/smoke_cg2_checkpoint.py" --input-json "$OUT/instruction.json" --output "$OUT/cg2_$N"
  singularity exec --cleanenv "$IMAGE" mpiexec -np "$N" "$FENICS_PYTHON" \
    "$REPO/validation/smoke_event_mpi.py" --expected-size "$N" --output "$OUT/event_$N.json"
done
printf '%s\n' 'Preflight scripts returned successfully; inspect all JSON reports before any LV smoke.'
