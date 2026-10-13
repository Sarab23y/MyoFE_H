#!/bin/bash
#SBATCH --job-name=myofe_validation
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=validation-smoke.%j.out

set -euo pipefail

: "${FENICS_IMAGE:?Set FENICS_IMAGE to the working legacy FEniCS image}"
: "${VOLUME_SCALE_TO_ML:?Set VOLUME_SCALE_TO_ML after verifying mesh units}"

REPO=${REPO:-$SLURM_SUBMIT_DIR}
RESULTS="$REPO/validation/results"
TMP=${SLURM_TMPDIR:-/tmp}
mkdir -p "$RESULTS/material" "$RESULTS/inflation_serial" \
         "$RESULTS/inflation_mpi"

singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python validation/verify_material_validation.py \
   validation/configs/material_validation.json"

make_config () {
  local output=$1
  local destination=$2
  python - "$REPO/validation/configs/passive_inflation.json" \
      "$destination" "$output" "$VOLUME_SCALE_TO_ML" <<'PY'
import json, sys
source, destination, output, scale = sys.argv[1:]
with open(source) as handle:
    data = json.load(handle)
data['pressures_mmHg'] = [0.0, 1.0]
data['output_directory'] = output
data['volume_scale_to_ml'] = float(scale)
with open(destination, 'w') as handle:
    json.dump(data, handle, indent=2)
PY
}

make_config "$RESULTS/inflation_serial" "$TMP/inflation_serial.json"
make_config "$RESULTS/inflation_mpi" "$TMP/inflation_mpi.json"

singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python validation/passive_lv_inflation.py \
   '$TMP/inflation_serial.json'"
mpiexec -np 2 singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python validation/passive_lv_inflation.py \
   '$TMP/inflation_mpi.json'"

singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python validation/compare_inflation_runs.py \
   '$RESULTS/inflation_serial/passive_inflation.csv' \
   '$RESULTS/inflation_mpi/passive_inflation.csv'"

# Plotting is deliberately explicit and will fail clearly if the selected
# image does not provide Matplotlib.
singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd '$REPO' && python postprocessing/plot_passive_inflation.py \
   '$RESULTS/inflation_mpi/passive_inflation.csv' --headless \
   --save '$RESULTS/inflation_mpi/passive_inflation.png'"
