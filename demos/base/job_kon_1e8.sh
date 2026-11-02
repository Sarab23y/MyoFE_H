#!/bin/bash
#SBATCH --time=12-00:00:00             # Time limit for the job (REQUIRED).
#SBATCH --job-name=kon_1e8
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1               # Number of CPUs (cores) per task.
#SBATCH --mem=180G                      # Memory required per node.
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L         # Partition/queue (REQUIRED).
#SBATCH --nodelist=skylake051
#SBATCH --output=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/kon_1e8/sim_output/output.%J.out
#SBATCH --error=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/kon_1e8/sim_output/output.%J.err

set -euo pipefail

# The Slurm log directory must exist before submitting this script because
# Slurm opens the --output/--error files before the job body starts.
cd /home/sba431/MyoFE_H/python_codes

singularity exec --cleanenv /home/sba431/fenics.img \
    mpiexec -np "$SLURM_NTASKS" python MyoFE.py LV_sim \
    /home/sba431/MyoFE_H/demos/base/sim_inputs/base_instruction_kon_1e8.json
