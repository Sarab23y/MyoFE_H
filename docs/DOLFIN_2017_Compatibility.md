# Authoritative DOLFIN 2017.1.0 compatibility — job 6354446

**Production runtime: DOLFIN 2017.1.0 in /home/sba431/fenics.img. Do not upgrade it.**

## Failure and correction

Job 6354446 stopped in native_and_cg2 at the builder's exact 2019.1.0 version guard. This intentional guard ran before observer compilation or the CG2 test. The recorded exit 134 does not establish a failed Newton solve, inverted element or abnormal physical deformation.

The observer requires protected virtual `NewtonSolver::converged(const GenericVector&, const NonlinearProblem&, size_t)` and `update_solution(GenericVector&, const GenericVector&, double, const NonlinearProblem&, size_t)`, plus native residual(), relative_residual() and krylov_iterations() accessors. It forwards each original method once, preserves its arguments and return, and never substitutes a Newton loop. The update callback reads the actual live displacement before/after the original update; trial F/J/stretch are separate diagnostic projections. The model continues to use its original high-level `solve(..., J=..., newton_solver={...}, representation=uflacs)` path in nsolver.py. A Python Newton subclass would not attach automatically to that internal solver, so no solver replacement is introduced.

The nearby official [2017.2 API](https://fenics.readthedocs.io/projects/dolfin/en/2017.2.0/apis/api_nls.html) documents virtual hook customization and those accessors. It is **not proof of the 2017.1 container binary or Python bindings**. The exact 2017.1 source/image was unavailable locally. Accordingly this update treats 2017.1 as an explicit candidate profile with mandatory checks against the actual installed headers and exported symbols, rather than claiming binary compatibility from adjacent documentation.

Changes:

- Explicit 2017.1.0 and preserved 2019.1.0 profiles; unknown versions rejected. The production preflight requires exactly 2017.1.0.
- On Linux x86_64, require exactly one exported hook with each exact demangled signature, including const/reference/relaxation/size_t layout. Ambiguous or changed overloads are rejected. unsigned-long size_t is explicitly restricted to this target ABI.
- Compile an `override`/static_assert contract against the actual image headers before linking. This verifies virtual signatures and accessor return types without instantiating/replacing the solver. Link with undefined-symbol rejection.
- Preserve library/source/hook hashes and compile command; archive generated contract and adapter profile. build.json explicitly leaves callback reachability and physical invariance false. A build success is not telemetry success.
- Replace Python-3-only check_output(input=...) with Popen.communicate for production Python 2 compatibility.
- Use existing world_comm/as_mpi4py in the independent checkpoint reader, avoiding newer-only MPI.comm_world and retaining the actual communicator for DOLFIN.
- Coordinate the native probe's deliberate last-rank error before sink reductions, so the failure fixture cannot strand peers in a collective.
- Record runtime version in native/CG2 reports. Default only diagnostic validation launchers to CAC48M192_L; exclude cascadeb024. No production launcher changed.

## Telemetry and safe failure

Candidate recorded quantities: native iteration and residual/relative residual, actual applied displacement correction L2/max (pressure/Real excluded in LV), relaxation, native convergence, cumulative linear iterations, trial global J extrema and minimum fiber stretch; accepted-state F/J/stretch are evaluated separately after the original solve. No per-iteration pointwise stresses or Jacobian condition numbers are claimed.

Header/symbol/link checks cannot prove ELF interposition, GIL re-entry, collective ordering or numerical invariance. The native manufactured probe must demonstrate callback sequence, direct correction agreement, F=I initial state, unchanged solution and coordinated failure at 1 and 32 ranks. The actual LV three-arm comparisons are still required. Missing callbacks or unsupported criteria fail the diagnostic; no invented values or silent unobserved fallback are reported as success. If forwarding is unavailable in the actual build, stop and report per-iteration telemetry unavailable. Do not replace production Newton to make the test pass.

## CG2 assessment

Writer and reader use legacy HDF5File(comm, native-string filename, mode), read/write(Function, path), read(Mesh, path, False), and close(), matching the repository's established HDF5 procedure. The original mixed CG2/CG1/Real/5-Real space and all mixed coefficients are written; CG1 visualization is never the reconstruction source. The manufactured field has nonzero CG2 edge information, exact same-space vector roundtrip is checked, and a separate mesh/space reader reconstructs F/J/stretch at coordinates. Actual image calls, HDF5 parallel support and cross-rank partition handling remain unexecuted. Those reports must pass at both 1 and 32 ranks.

## Other compatibility audit

Quadrature scalar/vector elements retain degree 2 and default scheme; diagnostic projections reuse the existing fenics_2017_compat helper and quadrature backend. Production forms/compilation remain unchanged. Mixed spaces retain the production element layout. Owned vectors, authoritative local/global IDs, geometry and rank scatter stay guarded by the existing actual-mesh validators. Python object collectives use mpi4py views, never a PETSc Comm directly. Checkpoint filenames use existing native-string conversion. Finalization/event buffers/source and input provenance code are not refactored; their existing runtime tests remain gates. No further version-specific defect was demonstrated locally, so no speculative physical changes are made.

## Exact narrow LCC test first

No LCC job was executed from Windows. These commands submit only the reviewed native/CG2 preflight; choose a fresh directory and keep old job reports.

```bash
cd /mnt/gpfs2_4m/scratch/sba431/MyoFE_H_30cycle_validation
git status --short --branch
# If detached or containing local changes, stop and inspect; do not reset/stash.
git fetch origin diagnostics/30cycle-fe-myosim-validation
# Fast-forward only when on the intended local branch:
git switch diagnostics/30cycle-fe-myosim-validation
git pull --ff-only origin diagnostics/30cycle-fe-myosim-validation
export REPO=$PWD
export IMAGE=/home/sba431/fenics.img
export ANALYSIS_PYTHON=/opt/ohpc/pub/libs/Miniconda3/bin/python
export BASELINE_INPUT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/instruction.json
export DATA=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/sim_output
# Use the original working directory used to resolve relative production mesh paths:
export BASELINE_REPO=/home/sba431/MyoFE_H
export ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/FE30_2017_probe_$(date -u +%Y%m%dT%H%M%SZ)
test ! -e "$ROOT" && mkdir "$ROOT"
"$ANALYSIS_PYTHON" -c 'import sys,numpy,h5py,matplotlib; assert sys.version_info[:3]==(3,7,10); print(sys.version)'
"$ANALYSIS_PYTHON" "$REPO/validation/prepare_fe_cycle_case.py"  --baseline-input "$BASELINE_INPUT" --baseline-repo "$BASELINE_REPO"  --template "$REPO/demos/ECM/sim_input/base_instruction_Br_ECM_30cycles_FE_diagnostic_smoke.json"  --repo "$REPO" --image "$IMAGE" --mode smoke --output "$ROOT/prepared"
export INPUT="$ROOT/prepared/instruction.json"
export OUT="$ROOT/native_cg2"
squeue -u "$USER"
JOB=$(sbatch --parsable --partition=CAC48M192_L --nodes=1 --ntasks=32  --exclude=cascadeb024 --export=ALL "$REPO/validation/run_fe_cycle_preflight.slurm")
echo "$JOB"
squeue -j "$JOB"
```

If the existing worktree is detached, keep it intact and create a fresh sibling worktree from origin/diagnostics/30cycle-fe-myosim-validation, then set REPO to that sibling. Do not switch a dirty worktree or assume BASELINE_REPO is correct for a relative mesh; preparation must find and hash the actual original mesh.

After job completion, require all files, not just a zero exit:

```bash
sacct -j "$JOB" --format=JobID,State,Elapsed,MaxRSS,ExitCode
cat "$OUT/model_import.log"
test ! -f "$OUT/failure.txt" || cat "$OUT/failure.txt"
cat "$OUT/native/build.json"
for N in 1 32; do
 cat "$OUT/native_$N/native_probe.json"
 cat "$OUT/cg2_$N/checkpoint_smoke.json"
 cat "$OUT/cg2_$N/reconstruction.json"
 cat "$OUT/event_$N.json"
done
```

## Complete short validation only after narrow probes pass

Keep REPO, IMAGE, ANALYSIS_PYTHON, BASELINE_INPUT, BASELINE_REPO and DATA as above. ROOT must be a new nonexistent directory; the complete launcher prepares its own inputs.

```bash
export ROOT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/FE30_2017_short_$(date -u +%Y%m%dT%H%M%SZ)
test ! -e "$ROOT"
SHORT_JOB=$(sbatch --parsable --partition=CAC48M192_L --nodes=1 --ntasks=32  --exclude=cascadeb024 --export=ALL "$REPO/validation/validate_lcc.slurm")
echo "$SHORT_JOB"
```

This reuses the existing complete short launcher (native/CG2, sealed mapping, three 200-step arms at both rank counts, independent HDF/XDMF integrity, intentional interruption and source-driver mock). No full simulation is submitted. See its ASSESSMENT.txt and invariance/integrity reports. Short pre-baroreflex runs cannot prove later-cycle behavior or actual 30-cycle termination.

**GO for rerunning the narrow compatibility probe, subject to normal LCC scheduling. NO-GO for the full 30-cycle run.** Actual native/CG2, 1/32-rank mapping, physical invariance and output-integrity evidence remain required. A changed signature/header/ABI or failed comparison is a blocker, not a reason to relax settings.
