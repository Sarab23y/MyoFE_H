#!/bin/bash
#SBATCH --job-name=FE30_validation_only
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L
#SBATCH --exclude=cascadeb024
#SBATCH --output=FE30_validation_%j.out
#SBATCH --error=FE30_validation_%j.err
# Diagnostic short cases only. No full-run mode, source edits or nested submissions.
set -Eeuo pipefail
: "${REPO:?Set existing isolated diagnostic worktree}"
: "${BASELINE_REPO:?Set original production worktree, read-only}"
: "${BASELINE_INPUT:?Set actual submitted baseline instruction}"
: "${DATA:?Set existing saved output with validated coordinate files}"
: "${ANALYSIS_PYTHON:?Set absolute Python3.7-3.12 with NumPy h5py Matplotlib}"
: "${ROOT:?Set NEW absolute output directory}"
IMAGE=${IMAGE:-/home/sba431/fenics.img}
export FENICS_PYTHON=python # Existing mapping launcher uses this exact image interpreter.
export IMAGE REPO BASELINE_REPO BASELINE_INPUT DATA ANALYSIS_PYTHON ROOT
export BASH_TEST_PYTHON="$ANALYSIS_PYTHON"
unset LD_PRELOAD EXPECT_INTERRUPTED
[[ ${SLURM_NTASKS:-0} == 32 && ${SLURM_JOB_NUM_NODES:-0} == 1 ]]
[[ $ROOT == /* && $ANALYSIS_PYTHON == /* ]]
REPO=$(realpath "$REPO"); BASELINE_REPO=$(realpath "$BASELINE_REPO")
[[ $REPO != "$BASELINE_REPO" && ! -e $ROOT ]]
[[ $(hostname -s) != cascadeb024 ]]
for FILE in "$BASELINE_INPUT" "$IMAGE"; do test -f "$FILE"; done
test -d "$DATA"
mkdir "$ROOT" # Parent must exist; refusal to reuse is intentional.
STAGE=initial_checks
trap 'rc=$?; printf "NO-GO\nstage=%s\nexit=%s\n" "$STAGE" "$rc" > "$ROOT/FAILED.txt"; exit "$rc"' ERR
run_stage() {
 STAGE=$1; shift
 printf '%s START %s\n' "$(date -u +%FT%TZ)" "$STAGE" >> "$ROOT/stages.log"
 "$@" > "$ROOT/${STAGE}.log" 2>&1
 printf '%s PASS_RETURN %s\n' "$(date -u +%FT%TZ)" "$STAGE" >> "$ROOT/stages.log"
}
"$ANALYSIS_PYTHON" -c 'import sys,numpy,h5py,matplotlib; assert (3,7)<=sys.version_info[:2]<(3,13); print(sys.version)' > "$ROOT/analysis_environment.txt"
git -C "$REPO" status --porcelain > "$ROOT/diagnostic_status_before.txt"
git -C "$BASELINE_REPO" status --porcelain > "$ROOT/production_status_before.txt"
sha256sum "$BASELINE_INPUT" "$IMAGE" > "$ROOT/original_inputs.sha256"
export TEMPLATE="$REPO/demos/ECM/sim_input/base_instruction_Br_ECM_30cycles_FE_diagnostic_smoke.json"
prepare() {
 "$ANALYSIS_PYTHON" "$REPO/validation/prepare_fe_cycle_case.py" \
  --baseline-input "$BASELINE_INPUT" --baseline-repo "$BASELINE_REPO" \
  --template "$TEMPLATE" --repo "$REPO" --image "$IMAGE" "$@"
}
run_stage prepare_probe prepare --mode smoke --output "$ROOT/preflight_case"
export INPUT="$ROOT/preflight_case/instruction.json"
export OUT="$ROOT/preflight"
# Imports, actual UFL/DOLFIN assembly, native forwarding and CG2 reopen at 1/32 ranks.
run_stage native_and_cg2 bash "$REPO/validation/run_fe_cycle_preflight.slurm"
export OUT="$ROOT/mapping"
# Existing suite includes scalar, mixed, coordinates, ownership/scatter and comparisons.
run_stage mapping_1_and_32 bash "$REPO/validation/run_phase_mapping_validation.slurm"
run_stage mapping_gate singularity exec --cleanenv "$IMAGE" python -c \
 'import sys; sys.path.insert(0,sys.argv[1]+"/validation"); from export_verified_quadrature_geometry import require_suite; require_suite(sys.argv[2],sys.argv[3],sys.argv[1]); print("All ten reports and hashes verified")' \
 "$REPO" "$OUT" "$OUT/instruction.json"
export COORDINATES="$ROOT/hotspot_coordinates.json"
STAGE=verified_coordinates
"$ANALYSIS_PYTHON" - <<'PY'
import os,json,numpy as np
ids=np.load(os.path.join(os.environ['DATA'],'quadrature_global_dof.npy')).reshape(-1)
xyz=np.load(os.path.join(os.environ['DATA'],'quadrature_dof.npy'))
assert xyz.shape==(len(ids),3) and len(set(ids.tolist()))==len(ids) and np.isfinite(xyz).all()
points=[]
for gid in (2833,4844,1580,2497):
 hit=np.flatnonzero(ids==gid); assert len(hit)==1, 'Missing/ambiguous historical ID %s'%gid
 points.append(xyz[int(hit[0])].tolist())
with open(os.environ['COORDINATES'],'x') as f:json.dump(points,f,indent=2)
PY
export NATIVE_HOOK="$ROOT/preflight/native/libmyofe_newton_observer.so"
for N in 1 32; do
 for MODE in pure-control control smoke; do
  run_stage "prepare_${MODE}_${N}" prepare --mode "$MODE" --coordinates-json "$COORDINATES" --output "$ROOT/${MODE}_${N}"
  export CASE="$ROOT/${MODE}_${N}/instruction.json"
  run_stage "run_${MODE}_${N}" env SLURM_NTASKS="$N" bash "$REPO/validation/run_fe_cycle_smoke.slurm"
 done
 run_stage "invariance_${N}" "$ANALYSIS_PYTHON" "$REPO/validation/compare_fe_cycle_smoke.py" \
  --pure-control "$ROOT/pure-control_${N}" --control "$ROOT/control_${N}" \
  --enabled "$ROOT/smoke_${N}" --output "$ROOT/invariance_${N}.json"
done
run_stage prepare_interruption prepare --mode smoke --coordinates-json "$COORDINATES" \
 --interrupt-after-steps 37 --output "$ROOT/interrupted_32"
export CASE="$ROOT/interrupted_32/instruction.json"
run_stage interruption_flush env EXPECT_INTERRUPTED=1 bash "$REPO/validation/run_fe_cycle_smoke.slurm"
# Source-driver mock only: NOT actual 30-cycle FE runtime verification.
run_stage termination_source_mock bash -c 'cd "$REPO"; "$ANALYSIS_PYTHON" -m unittest tests.test_fe_cycle_followup.IntegrityAndComparisonTests.test_original_driver_loop_stops_at_30_real_HR_intervals_and_flushes_closes -v'
STAGE=final_inventory
du -sk "$ROOT"/* > "$ROOT/disk_kib.txt"
df -h "$ROOT" > "$ROOT/filesystem.txt"
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocCPUS,ExitCode > "$ROOT/accounting_live.txt" || true
git -C "$BASELINE_REPO" status --porcelain > "$ROOT/production_status_after.txt"
cmp "$ROOT/production_status_before.txt" "$ROOT/production_status_after.txt"
git -C "$REPO" status --porcelain > "$ROOT/diagnostic_status_after.txt"
cmp "$ROOT/diagnostic_status_before.txt" "$ROOT/diagnostic_status_after.txt"
cat > "$ROOT/ASSESSMENT.txt" <<'EOF'
SHORT VALIDATION COMMANDS COMPLETED; FULL RUN: NO-GO pending scientific review.
Inspect native, CG2, ten mapping reports, invariance_1/32, all output_integrity
and interruption_integrity reports. PASS_RETURN denotes command success only.
Actual 30-cycle FE termination remains UNEXECUTED; only source-driver mock checked.
Early 200-step cases do not measure later-event performance or post-5s baroreflex.
No full simulation submitted. No model settings changed. No commit/push performed.
Measure completed-job sacct, output growth and event overhead before allocation.
EOF
printf 'Validation artifacts: %s\n' "$ROOT"
