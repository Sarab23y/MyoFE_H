#!/bin/bash
#SBATCH --job-name=myofe_growth_mass
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=growth-mass.%j.out
set -euo pipefail
: "${FENICS_IMAGE:?Set FENICS_IMAGE to the working legacy FEniCS image}"
REPO=${REPO:-$SLURM_SUBMIT_DIR}
for case in no_growth proportional collagen_enriched; do
  singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
    "cd '$REPO/python_codes' && python MyoFE.py LV_sim \
     ../validation/configs/growth_mass_validation_${case}.json"
done

singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python validation/check_growth_mass_results.py \
   validation/results/growth_mass/no_growth/data.csv \
   validation/results/growth_mass/proportional/data.csv \
   validation/results/growth_mass/collagen_enriched/data.csv"
