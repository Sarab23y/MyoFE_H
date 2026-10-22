#!/bin/bash
#SBATCH --time=12-00:00:00
#SBATCH --job-name=Br_ECM_stress
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --nodelist=skylake051
#SBATCH --output=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/ECM_stress_diagnostic.%J.out
#SBATCH --error=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/ECM_stress_diagnostic.%J.err
set -euo pipefail

REPO=/home/sba431/MyoFE_H
IMAGE=/home/sba431/fenics.img
JSON="$REPO/demos/base/sim_inputs/base_instruction_Br_ECM_stress_diagnostic.json"
RESULT=/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/ECM_stress_diagnostic
[ -f "$JSON" ] || { echo "ERROR: missing $JSON" >&2; exit 2; }
[ -f "$IMAGE" ] || { echo "ERROR: missing $IMAGE" >&2; exit 2; }

singularity exec --cleanenv "$IMAGE" python - "$JSON" <<'PY'
from __future__ import print_function
import json, sys
case=json.load(open(sys.argv[1])); p=case['protocol']; m=case['model']
first=lambda d,k:d[k][0]
assert (first(p,'time_step'),first(p,'no_of_time_steps')) == (0.001,60000)
assert tuple(first(p['baroreflex']['activations'][0],k) for k in ('t_start_s','t_stop_s')) == (5,60)
change=p['perturbation']['perturbations'][0]
assert tuple(first(change,k) for k in ('level','variable','t_start_s','t_stop_s','total_change')) == ('circulation','aorta_resistance',20,30,170)
d=m['ecm_stress_diagnostic']
assert first(d,'enabled') and first(d,'activation_time') == 20.0
assert first(d,'homeostatic_cycles') == 3 and not first(d,'composition_updates_enabled')
assert 'growth' not in m and 'fibrosis' not in m and not first(d,'geometric_growth_enabled')
print('PASS: 60-s ECM diagnostic configuration and 20-30-s +170 ramp verified')
PY

if [ -e "$RESULT" ]; then
  echo "ERROR: refusing to overwrite existing results: $RESULT" >&2
  exit 3
fi
mkdir -p "$RESULT/sim_output" "$RESULT/mesh_output"
cd "$REPO/python_codes"
singularity exec --cleanenv "$IMAGE" mpiexec -np "$SLURM_NTASKS" \
  python MyoFE.py LV_sim "$JSON" || { echo 'ERROR: MyoFE solver failed' >&2; exit 1; }
singularity exec --cleanenv "$IMAGE" python \
  "$REPO/postprocessing/plot_ecm_stress_diagnostic.py" \
  "$RESULT/sim_output/data.csv" "$RESULT/plots" || \
  { echo 'ERROR: ECM postprocessing failed' >&2; exit 1; }
echo "PASS: ECM diagnostic completed at $RESULT"
