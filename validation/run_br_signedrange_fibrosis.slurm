#!/bin/bash
#SBATCH --time=12-00:00:00
#SBATCH --job-name=Br_signedrange
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --nodelist=skylake044
#SBATCH --output=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/signedrange-tests.%J.out
#SBATCH --error=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/signedrange-tests.%J.err
set -euo pipefail

# The job intentionally cannot launch until measured, approved references are
# supplied in the same units as f0 . total_stress . f0.
: "${S_FF_MIN:?export an approved finite S_FF_MIN before sbatch}"
: "${S_FF_MAX:?export an approved finite S_FF_MAX before sbatch}"
REPO=${REPO:-/home/sba431/MyoFE_H}
FENICS_IMAGE=${FENICS_IMAGE:-/home/sba431/fenics.img}
RESULT_ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/signedrange_tests
CONFIG_DIR="$RESULT_ROOT/configs.${SLURM_JOB_ID}"
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$RESULT_ROOT" "$CONFIG_DIR"

# Preserve previous case results before reusing the documented output roots.
for case in control perturbed; do
  if [ -d "$RESULT_ROOT/$case" ]; then
    mv "$RESULT_ROOT/$case" "$RESULT_ROOT/${case}.backup.${STAMP}"
  fi
done

singularity exec --cleanenv "$FENICS_IMAGE" python \
  "$REPO/validation/verify_signedrange_configurations.py" \
  --allow-unresolved-bounds \
  "$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_signedrange_control.json" \
  "$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_signedrange_perturbed.json"

singularity exec --cleanenv "$FENICS_IMAGE" python \
  "$REPO/validation/prepare_stress_range_cases.py" \
  "$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_signedrange_control.json" \
  "$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_signedrange_perturbed.json" \
  "$S_FF_MIN" "$S_FF_MAX" "$CONFIG_DIR" "$RESULT_ROOT"

singularity exec --cleanenv "$FENICS_IMAGE" python \
  "$REPO/validation/verify_signedrange_configurations.py" \
  "$CONFIG_DIR/stress_range_control.json" \
  "$CONFIG_DIR/stress_range_perturbed.json"

for case in control perturbed; do
  mkdir -p "$RESULT_ROOT/$case/sim_output" "$RESULT_ROOT/$case/mesh_output"
  singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
    "cd '$REPO/python_codes' && mpiexec -np '${SLURM_NTASKS}' \
     python MyoFE.py LV_sim '$CONFIG_DIR/stress_range_${case}.json'"
done

singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python postprocessing/plot_fibrosis_validation.py \
   '$RESULT_ROOT/control/sim_output/data.csv' \
   '$RESULT_ROOT/perturbed/sim_output/data.csv' --headless \
   --save-directory '$RESULT_ROOT/plots.${SLURM_JOB_ID}'"
