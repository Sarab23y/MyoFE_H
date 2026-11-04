#!/bin/bash
#SBATCH --job-name=myofe_as_fibrosis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=as-fibrosis.%j.out
set -euo pipefail
: "${FENICS_IMAGE:?Set FENICS_IMAGE to the working legacy FEniCS image}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
for case in healthy aortic_stenosis; do
  singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
    "cd '$REPO/python_codes' && python MyoFE.py LV_sim \
     ../validation/configs/as_fibrosis_${case}.json"
done
singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python validation/check_as_fibrosis_results.py \
   validation/results/as_fibrosis/healthy/data.csv \
   validation/results/as_fibrosis/aortic_stenosis/data.csv"

singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python postprocessing/plot_fibrosis_validation.py \
   validation/results/as_fibrosis/healthy/data.csv \
   validation/results/as_fibrosis/aortic_stenosis/data.csv \
   --headless --save-directory validation/results/as_fibrosis/plots"
