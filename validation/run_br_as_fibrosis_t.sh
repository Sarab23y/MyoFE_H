#!/bin/bash
#SBATCH --time=12-00:00:00
#SBATCH --job-name=br_as_fib_g15
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --output=br-as-fib-g15.%j.out
#SBATCH --error=br-as-fib-g15.%j.err

# MyoFE_H: Healthy control + aortic stenosis fibrosis, gamma_c = 1.5.
# Uses the same container/MPI pattern as the existing working HPC job script.
set -euo pipefail

REPO="/home/sba431/MyoFE_H"
IMAGE="/home/sba431/fenics.img"
BASE="/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis"
CONTROL_JSON="$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_control.json"
AS_JSON="$REPO/demos/base/sim_inputs/base_instruction_Br_fibrosis_AS.json"
CONTROL_DIR="$BASE/AS_test_control"
AS_DIR="$BASE/AS_test"

for path in "$IMAGE" "$CONTROL_JSON" "$AS_JSON"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: Required file not found: $path" >&2
        exit 1
    fi
done

# Preserve results from earlier runs. Rename prior case folders before
# reusing the fixed output directories specified by the input JSON files.
TAG="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-manual}"
for case_dir in "$CONTROL_DIR" "$AS_DIR"; do
    if [ -d "$case_dir" ] && [ -n "$(ls -A "$case_dir")" ]; then
        backup="${case_dir}_backup_${TAG}"
        echo "Backing up previous output: $case_dir -> $backup"
        mv "$case_dir" "$backup"
    fi
done

mkdir -p "$CONTROL_DIR/sim_output" "$CONTROL_DIR/mesh_output"
mkdir -p "$AS_DIR/sim_output" "$AS_DIR/mesh_output"

cd "$REPO/python_codes"

echo "[$(date)] Starting HEALTHY control with gamma_c from JSON (expected 1.5)"
singularity exec --cleanenv "$IMAGE" \
    mpiexec -np "${SLURM_NTASKS:-32}" \
    python MyoFE.py LV_sim "$CONTROL_JSON"
echo "[$(date)] Healthy control completed"

echo "[$(date)] Starting AS + fibrosis with gamma_c from JSON (expected 1.5)"
singularity exec --cleanenv "$IMAGE" \
    mpiexec -np "${SLURM_NTASKS:-32}" \
    python MyoFE.py LV_sim "$AS_JSON"
echo "[$(date)] AS + fibrosis completed"

CONTROL_CSV="$CONTROL_DIR/sim_output/data.csv"
AS_CSV="$AS_DIR/sim_output/data.csv"
for path in "$CONTROL_CSV" "$AS_CSV"; do
    if [ ! -s "$path" ]; then
        echo "ERROR: Missing or empty simulation output: $path" >&2
        exit 1
    fi
done

cd "$REPO"
echo "[$(date)] Checking constituent balances and gamma outputs"
singularity exec --cleanenv "$IMAGE" python validation/check_as_fibrosis_results.py \
    "$CONTROL_CSV" "$AS_CSV" \
    --initial-collagen-fraction 0.026 --myofiber-fraction 0.700

mkdir -p "$AS_DIR/sim_output/plots"
echo "[$(date)] Generating comparison plots"
singularity exec --cleanenv "$IMAGE" python postprocessing/plot_fibrosis_validation.py \
    "$CONTROL_CSV" "$AS_CSV" --headless \
    --save-directory "$AS_DIR/sim_output/plots"

# Spatial plotting depends on which spatial CSV files were written.
if ! singularity exec --cleanenv "$IMAGE" python postprocessing/plot_spatial_collagen.py \
    "$AS_DIR/sim_output" --headless \
    --save "$AS_DIR/sim_output/plots/collagen_before_after.png"; then
    echo "WARNING: Spatial collagen plotting was not completed; simulation CSVs are intact" >&2
fi

echo "[$(date)] Finished. Results:"
echo "  Healthy: $CONTROL_CSV"
echo "  AS:      $AS_CSV"
echo "  Plots:   $AS_DIR/sim_output/plots"
