#!/bin/bash
#SBATCH --job-name=gamma15_fibrosis
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=12-00:00:00
#SBATCH --mem=180G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --nodelist=skylake051
#SBATCH --output=gamma15-fibrosis.%j.out
#SBATCH --error=gamma15-fibrosis.%j.err
set -euo pipefail
FENICS_IMAGE=${FENICS_IMAGE:-/home/sba431/fenics.img}
REPO=${REPO:-/home/sba431/MyoFE_H}
CONTROL_ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test_control
AS_ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test
STAMP=$(date +%Y%m%d-%H%M%S)

# Preserve any earlier result trees before the JSON paths are reused.
for root in "$CONTROL_ROOT" "$AS_ROOT"; do
  if [ -e "$root/sim_output" ]; then
    mv "$root/sim_output" "$root/sim_output.backup.$STAMP"
  fi
  mkdir -p "$root/sim_output/mesh_output"
done

singularity exec --cleanenv "$FENICS_IMAGE" python \
  "$REPO/validation/verify_severe_as_configuration.py" \
  "$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_control.json" \
  "$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_AS.json"

for case in control AS; do
  singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
    "cd '$REPO/python_codes' && mpiexec -np '${SLURM_NTASKS}' \
     python MyoFE.py LV_sim \
     '$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_${case}.json'"
done
CONTROL=$CONTROL_ROOT/sim_output
AS=$AS_ROOT/sim_output
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
