#!/bin/bash
#SBATCH --job-name=phase_mapping
#SBATCH --output=phase_mapping_%j.out
#SBATCH --error=phase_mapping_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L
#SBATCH --nodelist=cascadeb024
set -euo pipefail
REPO=${REPO:-/home/sba431/MyoFE_H_diagnostics}
IMAGE=${IMAGE:-/home/sba431/fenics.img}
DATA=${DATA:-/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/sim_output}
INPUT=${INPUT:-$(dirname "$DATA")/instruction.json}
OUT=${OUT:-/mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_mapping_${SLURM_JOB_ID}}
REPO=$(realpath "$REPO")
PRODUCTION_REPO=$(realpath -m /home/sba431/MyoFE_H)
if test "$REPO" = "$PRODUCTION_REPO"; then
  echo 'Refusing to run from the production checkout; set REPO to the diagnostic worktree.' >&2
  exit 2
fi
test "$(git -C "$REPO" rev-parse --show-toplevel)" = "$REPO"
test "${SLURM_NTASKS}" = 32
test -f "$INPUT"
test -f "$IMAGE"
test -d "$DATA"
test ! -e "$OUT"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
cd "$REPO/python_codes"
{
  hostname
  date -u
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" status --short
  sha256sum "$IMAGE" "$INPUT"
  singularity --version
  singularity exec --cleanenv "$IMAGE" python -c 'import sys,dolfin,numpy; print(sys.version); print(dolfin.__version__); print(numpy.__version__)'
  singularity exec --cleanenv "$IMAGE" mpiexec --version
} > "$OUT/environment.txt" 2>&1
cp "$INPUT" "$OUT/instruction.json"
# Every test consumes this frozen copy of the submitted input, not a mutable
# repository configuration. Mesh and input checksums are recorded per test.
INPUT="$OUT/instruction.json"
for N in 1 "$SLURM_NTASKS"; do
  singularity exec --cleanenv "$IMAGE" mpiexec -np "$N" python \
    "$REPO/validation/mpi_compat.py" --expected-size "$N" --output "$OUT/mpi_comm_${N}.json"
  singularity exec --cleanenv "$IMAGE" mpiexec -np "$N" python \
    "$REPO/validation/validate_ecm_quadrature_mpi.py" --output "$OUT/ecm_${N}.json"
  singularity exec --cleanenv "$IMAGE" mpiexec -np "$N" python \
    "$REPO/validation/validate_myosim_mixed_quadrature_ordering.py" \
    --input-json "$INPUT" --output "$OUT/mixed_${N}.json"
  EXTRA=()
  if test "$N" = "$SLURM_NTASKS"; then EXTRA=(--saved-output "$DATA"); fi
  singularity exec --cleanenv "$IMAGE" mpiexec -np "$N" python \
    "$REPO/validation/export_verified_quadrature_geometry.py" \
    --input-json "$INPUT" "${EXTRA[@]}" --output "$OUT/geometry_${N}.json"
done
singularity exec --cleanenv "$IMAGE" python "$REPO/validation/validate_ecm_quadrature_mpi.py" \
  --compare "$OUT/ecm_1.json" "$OUT/ecm_${SLURM_NTASKS}.json" --output "$OUT/ecm_comparison.json"
singularity exec --cleanenv "$IMAGE" python "$REPO/validation/export_verified_quadrature_geometry.py" \
  --compare "$OUT/geometry_1.json" "$OUT/geometry_${SLURM_NTASKS}.json" --output "$OUT/geometry_comparison.json"
singularity exec --cleanenv "$IMAGE" python "$REPO/validation/export_verified_quadrature_geometry.py" \
  --seal-suite "$OUT" --input-json "$INPUT" --image "$IMAGE" --repo "$REPO"
echo "All mapping checks passed. Reports: $OUT"
