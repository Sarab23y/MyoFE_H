#!/bin/bash
#SBATCH --time=12-00:00:00
#SBATCH --job-name=ECM_40s_prov
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=SKY32M192_L
#SBATCH --nodelist=skylake051
#SBATCH --mail-user=Sara.b@uky.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_40s_provisional.%J.out
#SBATCH --error=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_40s_provisional.%J.err
set -euo pipefail

REPO=/home/sba431/MyoFE_H
IMAGE=/home/sba431/fenics.img
JSON="$REPO/demos/ECM/sim_input/base_instruction_Br_ECM_40s_provisional.json"
RESULT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_provisional

[ -f "$JSON" ] || { echo "ERROR: missing $JSON" >&2; exit 2; }
[ -f "$IMAGE" ] || { echo "ERROR: missing $IMAGE" >&2; exit 2; }

singularity exec --cleanenv "$IMAGE" python - "$JSON" <<'PY'
from __future__ import print_function
import json, sys
case=json.load(open(sys.argv[1])); protocol=case['protocol']; model=case['model']
first=lambda d,k:d[k][0]
assert (first(protocol,'time_step'),first(protocol,'no_of_time_steps')) == (0.001,80000)
assert tuple(first(protocol['baroreflex']['activations'][0],k)
             for k in ('t_start_s','t_stop_s')) == (5,80)
change=protocol['perturbation']['perturbations']
assert len(change) == 1
change=change[0]
assert tuple(first(change,k) for k in
             ('level','variable','t_start_s','t_stop_s','total_change')) == \
       ('circulation','aorta_resistance',40,50,170)
diag=model['ecm_stress_diagnostic']
assert first(diag,'enabled') and first(diag,'activation_time') == 40.0
assert first(diag,'homeostatic_cycles') == 3
assert first(diag,'baseline_stability_relative_tolerance') == 0.05
assert first(diag,'baseline_stability_mode') == 'warn'
assert first(diag,'freeze_homeostatic_field')
assert not first(diag,'composition_updates_enabled')
assert not first(diag,'geometric_growth_enabled')
assert 'growth' not in model and 'fibrosis' not in model
passive=case['mesh']['forms_parameters']['passive_law_parameters']
assert tuple(first(passive,k) for k in ('phi_m','phi_g','phi_c')) == (0.700,0.274,0.026)
assert first(case['model']['circulation']['compartments']['aorta'],'resistance') == 20
print('PASS: provisional 80-s ECM diagnostic, 40-s calibration, and 40-50-s +170 ramp verified')
PY

if [ -e "$RESULT" ]; then
  echo "ERROR: refusing to overwrite existing results: $RESULT" >&2
  exit 3
fi
mkdir -p "$RESULT/sim_output" "$RESULT/mesh_output" "$RESULT/plots"

cd "$REPO/python_codes"
if ! singularity exec --cleanenv "$IMAGE" mpiexec -np "$SLURM_NTASKS" \
  python MyoFE.py LV_sim "$JSON"; then
  echo 'ERROR: MyoFE solver failed' >&2
  if [ -f "$RESULT/sim_output/data.csv" ]; then
    singularity exec --cleanenv "$IMAGE" python \
      "$REPO/validation/diagnose_ecm_baseline_stability.py" "$RESULT" \
      --activation-time 40 || true
  else
    echo 'Inspect sim_output/ecm_baseline_stability.json and this SLURM log.' >&2
  fi
  exit 1
fi

singularity exec --cleanenv "$IMAGE" python \
  "$REPO/postprocessing/plot_ecm_stress_diagnostic.py" \
  "$RESULT/sim_output/data.csv" "$RESULT/plots" \
  --baseline-end 40 --ramp-end 50 --simulation-end 80 || \
  { echo 'ERROR: ECM postprocessing failed' >&2; exit 1; }
echo "PASS: provisional ECM 40-s-baseline diagnostic completed at $RESULT"
