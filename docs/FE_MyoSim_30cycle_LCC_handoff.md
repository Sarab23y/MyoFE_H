# Staged LCC validation handoff — unexecuted

These commands are prepared for review, not submitted. No actual FE/MPI or physical-invariance PASS is claimed. Use fresh directories. Do not alter the production checkout or globally export LD_PRELOAD. Actual image Python must import the unchanged model; do not silently translate production code or switch environments if it fails.

## Install the review patch

On an isolated worktree at b5d051a14433c48257e1f6797863592dfd4d6889, use `git apply --check cumulative.patch` then `git apply cumulative.patch`. If the old 15-file prototype is already present, use **incremental.patch instead**, never both. Stop on differences; do not reset or overwrite local work. No commit/push is required for validation. Compare the actual production/archive source with this base before making historical-source claims; matching input JSON alone is insufficient.

## Explicit environment and authenticated baseline

```bash
export REPO=/home/sba431/MyoFE_H_30cycle_diagnostics
export BASELINE_REPO=/home/sba431/MyoFE_H
export IMAGE=/home/sba431/fenics.img
# Set these to actual archived submitted input and existing output, not a guessed template:
export BASELINE_INPUT=/absolute/path/to/actual/submitted/instruction.json
export DATA=/absolute/path/to/existing/healthy/sim_output
export ANALYSIS_PYTHON=/absolute/path/to/python3
export BASH_TEST_PYTHON="$ANALYSIS_PYTHON"
export FENICS_PYTHON=python  # must be the working production interpreter inside this image
"$ANALYSIS_PYTHON" -c 'import sys,numpy,h5py,matplotlib; print(sys.version)'
# Analysis Python 3.7–3.12 recommended; test-only lib2to3 is unavailable in 3.13+.
test "$REPO" != "$BASELINE_REPO"
export ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/FE30_followup_$(date -u +%Y%m%dT%H%M%SZ)
test ! -e "$ROOT" && mkdir "$ROOT"
export TEMPLATE="$REPO/demos/ECM/sim_input/base_instruction_Br_ECM_30cycles_FE_diagnostic_smoke.json"
"$ANALYSIS_PYTHON" "$REPO/validation/prepare_fe_cycle_case.py"  --baseline-input "$BASELINE_INPUT" --baseline-repo "$BASELINE_REPO"  --template "$TEMPLATE" --repo "$REPO" --image "$IMAGE"  --mode smoke --output "$ROOT/preflight_case"
export INPUT="$ROOT/preflight_case/instruction.json"
```

Preparation resolves the exact original production mesh path to an absolute filename and records its hash, source hashes, baseline input and image hashes. It rejects LFS pointers and a manual/debug Newton branch. This input has a short-run cap and separate outputs; it retains the baseline model and mesh parameters. Inspect provenance before proceeding.

## Existing mapping suite, then native/CG2 probes

After approval for these short diagnostic jobs, check node availability and the potentially active production job. Existing mapping launcher pins skylake047; defer if busy. Exclude cascadeb024 and never modify the production launcher.

```bash
squeue -u "$USER" -o '%.18i %.9P %.24j %.8T %.10M %.20R'
sinfo -N -n skylake047,cascadeb024
export OUT="$ROOT/mapping"
MAP_JOB=$(sbatch --parsable --export=ALL --exclude=cascadeb024 "$REPO/validation/run_phase_mapping_validation.slurm")
echo "$MAP_JOB"
squeue -j "$MAP_JOB"
# After completion: inspect all ten reports and mapping_manifest.json, not just exit status.
cat "$OUT/failure.txt" 2>/dev/null || true
cat "$OUT/mapping_manifest.json"
```

Do not start the next stage until all scalar, mixed-state, geometry, MPI communicator and saved-coordinate checks at both rank counts pass. The manifest must bind this exact input and source/mesh provenance. If historical mapping files are missing, stop and recover them; do not infer identities by column position.

```bash
export OUT="$ROOT/preflight"
PREFLIGHT_JOB=$(sbatch --parsable --partition=SKY32M192_D --exclude=cascadeb024 --export=ALL "$REPO/validation/run_fe_cycle_preflight.slurm")
echo "$PREFLIGHT_JOB"
# After completion:
cat "$OUT/model_import.log"
cat "$OUT/native/build.json"
for N in 1 32; do
 cat "$OUT/native_$N/native_probe.json"
 cat "$OUT/cg2_$N/checkpoint_smoke.json"
 cat "$OUT/cg2_$N/reconstruction.json"
 cat "$OUT/event_$N.json"
done
export NATIVE_HOOK="$OUT/native/libmyofe_newton_observer.so"
```

The preflight imports the actual model, compiles against DOLFIN 2019.1.0, then runs manufactured native-observer, original-CG2 checkpoint and MPI event-union tests at 1 and 32 ranks. A successful manufactured probe is not LV invariance or historical deformation evidence. Any ABI, import or missing callback failure blocks later stages.

## Verified physical hotspots and three-arm LV comparisons

Only after historical coordinate mapping passes, create the four coordinate triples in the configured order from explicit saved IDs:

```bash
export COORDINATES="$ROOT/hotspot_coordinates.json"
"$ANALYSIS_PYTHON" - <<'PY'
import os,json,numpy as np
p=os.environ['DATA']
ids=np.load(os.path.join(p,'quadrature_global_dof.npy')).reshape(-1)
xyz=np.load(os.path.join(p,'quadrature_dof.npy'))
assert xyz.shape==(len(ids),3) and len(set(ids.tolist()))==len(ids)
points=[]
for gid in (2833,4844,1580,2497):
 hit=np.flatnonzero(ids==gid)
 assert len(hit)==1, 'Missing/ambiguous historical ID %s'%gid
 points.append(xyz[int(hit[0])].tolist())
with open(os.environ['COORDINATES'],'x') as f:json.dump(points,f,indent=2)
PY
for N in 1 32; do
 for MODE in pure-control control smoke; do
  "$ANALYSIS_PYTHON" "$REPO/validation/prepare_fe_cycle_case.py"    --baseline-input "$BASELINE_INPUT" --baseline-repo "$BASELINE_REPO"    --template "$TEMPLATE" --repo "$REPO" --image "$IMAGE"    --coordinates-json "$COORDINATES" --mode "$MODE" --output "$ROOT/${MODE}_${N}"
 done
done
```

Submit each triplet only after reviewing all preceding reports. All three arms run sequentially on the same allocated node. Allocations are provisional ceilings; do not launch all stages blindly.

```bash
for N in 1 32; do
 sbatch --parsable --nodes=1 --ntasks="$N" --cpus-per-task=1 --mem=180G   --time=02:00:00 --account=col_jfwe223_uksr --partition=SKY32M192_D   --exclude=cascadeb024 --export=ALL --output="$ROOT/invariance_${N}_%j.out"   --error="$ROOT/invariance_${N}_%j.err" --wrap='set -euo pipefail
N=$SLURM_NTASKS
for MODE in pure-control control smoke; do
 export CASE="$ROOT/${MODE}_${N}/instruction.json"
 bash "$REPO/validation/run_fe_cycle_smoke.slurm"
done
"$ANALYSIS_PYTHON" "$REPO/validation/compare_fe_cycle_smoke.py"  --pure-control "$ROOT/pure-control_${N}" --control "$ROOT/control_${N}"  --enabled "$ROOT/smoke_${N}" --output "$ROOT/invariance_${N}.json"'
done
```

Inspect exact initial-state equality before trajectory differences. The minimal witness records equilibrium active coefficient, native Newton telemetry and actual cellular/FE populations at every step; it is not an unobserved arm. Pure-control supplies the truly disabled path with common initial/final readouts. Pressure/volume and all saved fields use explicit timestamps. Every identity/count mismatch or numerical discrepancy blocks readiness; no tolerance widening.

## Flush failure path and event-specific mechanical evidence

After normal smokes pass, prepare a separate controlled interruption:

```bash
"$ANALYSIS_PYTHON" "$REPO/validation/prepare_fe_cycle_case.py"  --baseline-input "$BASELINE_INPUT" --baseline-repo "$BASELINE_REPO"  --template "$TEMPLATE" --repo "$REPO" --image "$IMAGE"  --coordinates-json "$COORDINATES" --mode smoke --interrupt-after-steps 37  --output "$ROOT/interrupted_32"
export CASE="$ROOT/interrupted_32/instruction.json"
export EXPECT_INTERRUPTED=1
sbatch --parsable --partition=SKY32M192_D --exclude=cascadeb024 --export=ALL "$REPO/validation/run_fe_cycle_smoke.slurm"
unset EXPECT_INTERRUPTED
# Require exact intentional boundary, metadata complete=false and finalized buffers:
cat "$ROOT/interrupted_32/interruption_integrity.json"
```

Normal enabled smokes create hash-bound first-frame reopen reports at 1/32 ranks, event chronology tables/plots and output_integrity.json. When actual events exist, separately reopen each event-adjacent original CG2 checkpoint with `validate_fe_cycle_checkpoint.py --state ABSOLUTE_STATE.h5 --output NEW_REPORT.json` under the same container MPI invocation used in the smoke launcher, at 1 and 32 ranks. Compare independent F/J/stretch at validated coordinates; first-frame reopening alone does not prove an event.

Measure `sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,AllocCPUS,ExitCode`, `du -sh "$ROOT"/*`, and `df -h "$ROOT"`. Initial 200-step smokes may contain no contraction/event and cannot establish long-run event overhead. Review storage_estimate.json and measured dimensions/iterations. Native JSON, original logs, postprocessing and backend RSS are extra categories. Stop before any full run: actual readiness evidence and user approval remain required.
