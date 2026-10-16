#!/bin/bash
#SBATCH --job-name=br_as_fibrosis
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=br-as-fibrosis.%j.out
set -euo pipefail
: "${FENICS_IMAGE:?Set FENICS_IMAGE to the working legacy FEniCS image}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
for case in control AS; do
  mpiexec -np "${SLURM_NTASKS}" singularity exec --cleanenv \
    "$FENICS_IMAGE" bash -lc \
    "cd '$REPO/python_codes' && python MyoFE.py LV_sim \
     ../demos/base/sim_inputs/base_instruction_Br_fibrosis_${case}.json"
done
CONTROL=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test_control/sim_output
AS=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test/sim_output
singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python validation/check_as_fibrosis_results.py \
   '$CONTROL/data.csv' '$AS/data.csv' \
   --initial-collagen-fraction 0.026 --myofiber-fraction 0.700"
singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python postprocessing/plot_fibrosis_validation.py \
   '$CONTROL/data.csv' '$AS/data.csv' --headless \
   --save-directory '$AS/plots'"
singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python postprocessing/plot_spatial_collagen.py \
   '$AS' --headless --save '$AS/plots/collagen_before_after.png'"
