#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=ECM_disp_cycles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L
#SBATCH --nodelist=cascadeb022
#SBATCH --output=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_disp_cycles.%J.out
#SBATCH --error=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_disp_cycles.%J.err
set -euo pipefail
REPO=${REPO:-/home/sba431/MyoFE_H}
PYTHON_BIN=${PYTHON_BIN:-python3}
BASE=${BASE:-/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_healthy_quad_validation_optimized}
XDMF=${XDMF:-$BASE/mesh_output/solution.xdmf}
QUADRATURE=${QUADRATURE:-$BASE/sim_output/quadrature_dof.npy}
OUT=${OUT:-/mnt/gpfs2_4m/scratch/sba431/demos/ECM/displacement_cycle_analysis_${SLURM_JOB_ID}}
test -f "$XDMF"; test -f "$QUADRATURE"; test ! -e "$OUT"
mkdir -p "$(dirname "$OUT")" "$REPO/demos/ECM/logs"
"$PYTHON_BIN" -c 'import numpy, h5py'
"$PYTHON_BIN" "$REPO/validation/compare_cycle_displacements.py" \
  --xdmf "$XDMF" --quadrature-coordinates "$QUADRATURE" --output "$OUT"
echo "Reports: $OUT"
