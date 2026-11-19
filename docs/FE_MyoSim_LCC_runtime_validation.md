> Production correction: DOLFIN **2017.1.0** is authoritative. Start with [the compatibility probe and exact LCC paths](DOLFIN_2017_Compatibility.md) before this complete short validation. Historical 2019 assumptions are superseded.

# One-launcher LCC validation handoff

**Current full-run assessment: NO-GO. Actual LCC/FEniCS checks are UNEXECUTED.**

The local implementation was not rewritten or edited. The review ZIP was read and its CRC and artifact-manifest hashes verified. No configured SSH target was available, and Windows has no SLURM/container runtime. No authentication was attempted against a guessed host. No job, commit or push was made.

## Before submission

Use the existing isolated diagnostic worktree with the reviewed cumulative OR incremental patch installed as described in the prior handoff. Do not apply both or modify production. The launcher is a new standalone artifact, not a replacement for any existing launcher. Transfer it to a new filename on LCC. All commands below run on LCC; replace the three explicit placeholders with actual paths.

```bash
export REPO=/mnt/gpfs2_4m/scratch/sba431/MyoFE_H_30cycle_validation
export BASELINE_REPO=/home/sba431/MyoFE_H
export BASELINE_INPUT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/instruction.json
export DATA=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/sim_output
export ANALYSIS_PYTHON=/opt/ohpc/pub/libs/Miniconda3/bin/python
export IMAGE=/home/sba431/fenics.img
export ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/FE30_runtime_$(date -u +%Y%m%dT%H%M%SZ)
export LAUNCHER="$REPO/validation/validate_lcc.slurm"
# Inspect the job queue first. No production node is reserved by this script.
squeue -u "$USER" -o '%.18i %.9P %.24j %.8T %.10M %.20R'
sinfo -N -p CAC48M192_L
"$ANALYSIS_PYTHON" -c 'import sys,numpy,h5py,matplotlib; print(sys.version)'
bash -n "$LAUNCHER"
test ! -e "$ROOT"
JOB=$(sbatch --parsable --export=ALL "$LAUNCHER")
printf 'Validation job: %s
Output root: %s
' "$JOB" "$ROOT"
squeue -j "$JOB"
```

The account/partition match the reviewed mapping workflow. The allocation is one node, 32 ranks, 180 GB, 8 hours, excluding cascadeb024. These are provisional limits, not measured requirements. It does not pin skylake047; nested existing launchers are executed with Bash inside this allocation, so their SBATCH directives do not submit jobs or override the allocation. Serial stages use one rank within the same allocated node. All six LV arms run sequentially on that node. No global LD_PRELOAD is retained. No full-run case is prepared or launched.

Use Python 3.7–3.12 with NumPy, h5py and Matplotlib for analysis/mock tests. The existing mapping workflow uses `python` inside the image; the wrapper consistently uses that interpreter and stops if it cannot import/compile the unchanged production model. Do not silently translate source or change the image. All input, mesh and source provenance checks remain active. Verify archived source parity with the actual production source before claiming historical fidelity; no source is copied into production.

## Ordered checks and acceptance

1. Prepare a separate capped input from the actual baseline; hash image, input, source and mesh.
2. Existing preflight: actual model imports, native C++ compilation, UFL/DOLFIN assembly, native-observer probe, CG2 edge-coefficient roundtrip and independent reconstruction at 1/32 ranks, and MPI event-union probes.
3. Existing mapping suite: one-rank then 32-rank scalar/mixed/geometry checks and cross-rank comparisons. `require_suite` verifies all ten reports, actual saved-coordinate validation, ownership/scatter and report/source/input hashes before hotspots are selected. Missing mapping data stops the job.
4. At each rank count, prepare and run pure-control, minimal witness-control and full short diagnostic arms, 200 native steps each. Existing launchers reopen actual CG2 HDF, validate XDMF references/times/flushes, and generate event reports. Exact initial state must match. Physical comparisons use atol=1e-10/rtol=1e-9; Newton residual/correction comparisons use atol=1e-12/rtol=1e-9; identity, iteration and status metadata must match exactly.
5. Run a separate 37-step intentional interruption and require the exact failure reason, incomplete-run metadata and final buffer completion checks.
6. Run the existing source-driver 30-cycle termination mock. This is NOT actual 30-cycle FE runtime verification.

The wrapper refuses reuse of ROOT and existing cases refuse output reuse. Do not resume a failed ROOT by deleting markers; inspect the failure and choose a fresh directory. The source files and scientific parameters remain unchanged. No historical output is written.

## Inspect after completion

```bash
sacct -j "$JOB" --format=JobID,State,Elapsed,MaxRSS,AllocCPUS,ExitCode
cat "$ROOT/stages.log"
# On failure, this names the stage. Do not continue until it is resolved.
test ! -f "$ROOT/FAILED.txt" || cat "$ROOT/FAILED.txt"
cat "$ROOT/preflight/native/build.json"
for N in 1 32; do
 cat "$ROOT/preflight/native_$N/native_probe.json"
 cat "$ROOT/preflight/cg2_$N/checkpoint_smoke.json"
 cat "$ROOT/preflight/cg2_$N/reconstruction.json"
 cat "$ROOT/invariance_$N.json"
 for MODE in pure-control control smoke; do
  cat "$ROOT/${MODE}_${N}/output_integrity.json"
 done
done
cat "$ROOT/mapping/mapping_manifest.json"
cat "$ROOT/interrupted_32/interruption_integrity.json"
cat "$ROOT/disk_kib.txt"
cat "$ROOT/ASSESSMENT.txt"
```

`PASS_RETURN` in stages.log means a command returned successfully, not full scientific certification. Physical/Newton differences are in invariance_1/32.json; file-content and FE-reopen evidence is in each integrity/reconstruction report. No differences have been measured locally. Missing reports, nonzero differences outside tolerance, absent native callbacks or an unrelated interruption failure are blockers. Actual event-adjacent checkpoints need separate reconstruction if events occur; first-frame reopening alone does not establish actual shortening.

## Resources and GO/NO-GO

The previous partial planning envelopes remain: about 177 GiB routine payload at the 60,001-step cap, 3.73 TiB for persistent triggering, and a separate 560 GiB native JSON envelope assuming 50 Newton iterations and 32 ranks. Original logs/postprocessing/RSS are additional. The 0.89-GiB global rolling-array envelope excludes backend/Python objects. There is no measured runtime, peak RSS, output rate or full-run allocation yet. Use completed-job accounting and actual output bytes; early 200-step cases cannot establish later-event costs or behavior after baroreflex activates at 5 s.

Even if all short checks pass, the wrapper leaves full-run NO-GO pending review: actual 30-cycle termination remains unexecuted, full-run resources need measurement, and later-cycle/event behavior is untested. No command to submit the full 30-cycle simulation is included. The existing full entry point remains blocked. Requesting a complete 30-cycle verification would require a separately approved runtime stage; a mock must not be represented as that evidence.
