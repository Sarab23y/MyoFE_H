#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=ECM_quad_check
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --nodelist=skylake051
#SBATCH --output=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_quad_check.%J.out
#SBATCH --error=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_quad_check.%J.err
set -euo pipefail
REPO=/home/sba431/MyoFE_H
IMAGE=/home/sba431/fenics.img
OUT=${OUT:-/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ecm_quadrature_validation_${SLURM_JOB_ID}}
mkdir -p "$OUT"
mpiexec -np 1 singularity exec --cleanenv "$IMAGE" python \
  "$REPO/validation/validate_ecm_quadrature_mpi.py" --output "$OUT/one_rank.json"
mpiexec -np 32 singularity exec --cleanenv "$IMAGE" python \
  "$REPO/validation/validate_ecm_quadrature_mpi.py" --output "$OUT/thirty_two_rank.json"
singularity exec --cleanenv "$IMAGE" python \
  "$REPO/validation/validate_ecm_quadrature_mpi.py" \
  --compare "$OUT/one_rank.json" "$OUT/thirty_two_rank.json" \
  --output "$OUT/comparison.json"
