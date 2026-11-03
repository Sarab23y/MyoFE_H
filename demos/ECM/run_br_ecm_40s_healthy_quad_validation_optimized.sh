#!/bin/bash
#SBATCH --time=12-00:00:00
#SBATCH --job-name=ECM_healthy_opt
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L
#SBATCH --nodelist=cascadeb022
#SBATCH --mail-user=Sara.b@uky.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_healthy_opt.%J.out
#SBATCH --error=/home/sba431/MyoFE_H/demos/ECM/logs/ECM_healthy_opt.%J.err
set -euo pipefail
REPO=/home/sba431/MyoFE_H
IMAGE=/home/sba431/fenics.img
JSON="$REPO/demos/ECM/sim_input/base_instruction_Br_ECM_40s_healthy_quad_validation_optimized.json"
RESULT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_healthy_quad_validation_optimized
[ -f "$JSON" ] && [ -f "$IMAGE" ] || { echo 'ERROR: missing JSON or image' >&2; exit 2; }
[ -n "${ECM_QUAD_VALIDATION_REPORT:-}" ] && \
  [ -f "$ECM_QUAD_VALIDATION_REPORT" ] || {
    echo 'ERROR: set ECM_QUAD_VALIDATION_REPORT to a passing comparison.json from run_ecm_quadrature_mpi.slurm' >&2
    exit 2
  }
singularity exec --cleanenv "$IMAGE" python - "$ECM_QUAD_VALIDATION_REPORT" <<'PY'
from __future__ import print_function
import json,sys
report=json.load(open(sys.argv[1]))
assert report['mode'] == 'cross_mpi_comparison' and report['passed']
print('PASS: prerequisite one-rank/32-rank quadrature comparison')
PY
singularity exec --cleanenv "$IMAGE" python - "$JSON" <<'PY'
from __future__ import print_function
import json,sys
case=json.load(open(sys.argv[1])); p=case['protocol']; m=case['model']
assert p['time_step'][0] == .001 and p['no_of_time_steps'][0] == 42000
assert p['perturbation']['perturbations'] == []
assert p['baroreflex']['activations'][0]['t_start_s'][0] == 5
assert p['baroreflex']['activations'][0]['t_stop_s'][0] == 42
assert m['ecm_stress_diagnostic']['activation_time'][0] == 40
assert m['ecm_stress_diagnostic']['homeostatic_cycles'][0] == 3
assert m['ecm_stress_diagnostic']['save_reference_cycle_snapshot'][0]
assert m['mechanical_health_diagnostic']['enabled'][0]
o=case['output_handler']; assert o['output_mode'][0] == 'buffered'
assert o['output_batch_size'][0] == 100
assert o['global_csv_frequency_n'][0] == 10
assert o['spatial_csv_frequency_n'][0] == 10
assert o['mesh_frequency_n'][0] == 100
assert o['output_profiling'][0]
assert 'growth' not in m and 'fibrosis' not in m
print('PASS: healthy-only 42-s quadrature validation configuration')
PY
[ ! -e "$RESULT" ] || { echo "ERROR: refusing to overwrite $RESULT" >&2; exit 3; }
mkdir -p "$RESULT/sim_output" "$RESULT/mesh_output" "$RESULT/plots"
cd "$REPO/python_codes"
singularity exec --cleanenv "$IMAGE" mpiexec -np "$SLURM_NTASKS" \
  python MyoFE.py LV_sim "$JSON"
