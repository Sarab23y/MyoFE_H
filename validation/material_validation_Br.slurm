#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=Material_Br
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --output=/mnt/gpfs2_4m/scratch/sba431/material_validation_Br/material.%j.out
#SBATCH --error=/mnt/gpfs2_4m/scratch/sba431/material_validation_Br/material.%j.err

set -euo pipefail

REPO_DIR=/home/sba431/MyoFE_H
FENICS_IMAGE=/home/sba431/fenics.img
CONFIG_FILE="$REPO_DIR/validation/configs/material_validation.json"

cd "$REPO_DIR"

export OMP_NUM_THREADS=1

# Expose both the repository and scratch output directory inside the container.
CONTAINER_BINDS="/home/sba431,/mnt/gpfs2_4m/scratch/sba431"

echo "Checking material formulation"

singularity exec --cleanenv \
    --bind "$CONTAINER_BINDS" \
    "$FENICS_IMAGE" \
    python "$REPO_DIR/validation/verify_material_validation.py" \
    "$CONFIG_FILE"

echo "Generating biaxial results"

singularity exec --cleanenv \
    --bind "$CONTAINER_BINDS" \
    "$FENICS_IMAGE" \
    python "$REPO_DIR/validation/biaxial_material_test.py" \
    "$CONFIG_FILE"

echo "Generating shear results"

singularity exec --cleanenv \
    --bind "$CONTAINER_BINDS" \
    "$FENICS_IMAGE" \
    python "$REPO_DIR/validation/shear_material_test.py" \
    "$CONFIG_FILE"

echo "Material verification and data generation completed."
