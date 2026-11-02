#!/bin/bash
#SBATCH --job-name=stress_range_fibrosis
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=12-00:00:00
#SBATCH --mem=180G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --nodelist=skylake051
#SBATCH --output=stress-range-fibrosis.%j.out
#SBATCH --error=stress-range-fibrosis.%j.err
set -euo pipefail

# These bounds must be approved measurements in the same stress units and
# sign convention as f0 . total_stress . f0. The job refuses placeholders.
: "${S_FF_MIN:?export an approved finite S_FF_MIN before sbatch}"
: "${S_FF_MAX:?export an approved finite S_FF_MAX before sbatch}"
FENICS_IMAGE=${FENICS_IMAGE:-/home/sba431/fenics.img}
REPO=${REPO:-/home/sba431/MyoFE_H}
RESULT_ROOT=${RESULT_ROOT:-/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/stress_range_${SLURM_JOB_ID}}
CONFIG_DIR="$RESULT_ROOT/configs"
mkdir -p "$CONFIG_DIR"

singularity exec --cleanenv "$FENICS_IMAGE" python \
  "$REPO/validation/prepare_stress_range_cases.py" \
  "$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_control.json" \
  "$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_AS.json" \
  "$S_FF_MIN" "$S_FF_MAX" "$CONFIG_DIR" "$RESULT_ROOT"

for case in control perturbed; do
  mkdir -p "$RESULT_ROOT/$case/sim_output" \
           "$RESULT_ROOT/$case/mesh_output"
  singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
    "cd '$REPO/python_codes' && mpiexec -np '${SLURM_NTASKS}' \
     python MyoFE.py LV_sim '$CONFIG_DIR/stress_range_${case}.json'"
done

singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python postprocessing/plot_fibrosis_validation.py \
   '$RESULT_ROOT/control/sim_output/data.csv' \
   '$RESULT_ROOT/perturbed/sim_output/data.csv' --headless \
   --save-directory '$RESULT_ROOT/plots'"
