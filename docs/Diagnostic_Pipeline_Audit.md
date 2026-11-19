# Diagnostic pipeline audit and preflight

This repair starts from `a0c9092b93684205549676c9ae97a04f7524d63b` on
`diagnostics/phase-regional-investigation`. Target execution is DOLFIN 2019.1.0
inside `/home/sba431/fenics.img`, with 32 ranks on `skylake047` in
`SKY32M192_D`. The production checkout is `/home/sba431/MyoFE_H` and the
diagnostic worktree is `/home/sba431/MyoFE_H_diagnostics`.

For the filename conversion failure in job 6340723 and the CAL/cascade001 rerun,
see [HDF5_Path_Compatibility.md](HDF5_Path_Compatibility.md). The preflight now
opens the actual mesh and wall dataset before the numerical validator stages,
logging the filename and communicator Python types at the constructor boundary.

## Issues identified and repaired

1. Standalone scripts attempted relative imports and caught only ImportError.
   Python 2 can raise ValueError in this context, as job 6340522 demonstrated.
   Every intra-diagnostic import now branches explicitly on `__package__`.
   Direct scripts use sibling imports; package modules use relative imports.
   Dependency failures propagate instead of being hidden by a fallback import.
2. Earlier tests exercised package imports but missed the exact launcher entry
   points. New subprocess tests invoke every mapping CLI directly, from another
   working directory, and with `python -m`. A regression test emulates the
   Python-2 relative-import exception. This supplements real target-interpreter
   checks in the preflight, rather than claiming a local Python-2 runtime test.
3. Mixed-state dimension discovery imported the cellular package, membranes and
   kinetics. It now derives the dimension from the submitted MyoSim bin grid
   and supported 3/4-state scheme, matching the production setup formula.
   Tests execute the original setup method bodies and compare dimensions.
   The diagnostic no longer needs pandas/SciPy/kinetics imports for this step.
4. Root-only report failures could strand other ranks at a later collective.
   `diagnostic_io.root_call` broadcasts the root's result or error to all ranks.
   Scalar, mixed, geometry and communicator reports use it at a common stage.
   HDF readers close their handles on failure. Exceptions inside MPI/DOLFIN
   native collective code can still require MPI's error handling or scheduler
   termination; the helper cannot make arbitrary native failures recoverable.
5. Reports and the completion manifest could overwrite existing files.
   Shared report writing now creates files exclusively, rejects nonfinite JSON,
   and removes its own incomplete file on serialization failure. The mixed
   failure handler also preserves an existing report. New run directories are
   created exclusively and interrupted suites retain partial reports/logs.
6. Comparison code could accept empty or failed upstream scalar reports.
   It now requires upstream success, nonempty matching points, finite values
   and matching direct-field names. Duplicate-coordinate detection uses counts.
   Geometry comparisons reject invalid weights/wall values. Tolerances and
   all quadrature/stress expressions are preserved.
7. A mesh could be selected from the caller's working directory. Resolution now
   follows the production convention, relative to `REPO/python_codes`, or an
   explicit absolute path. A missing mesh produces an actionable path error.
8. The optional displacement comparator treated a global DOF as an array row.
   It now requires the saved global-ID array and maps coordinates explicitly.
   It loads displacement histories only in the requested cycle interval.
   Its four-node visualization gradients remain approximations; original CG2
   coefficients and reference fibers are required for exact deformation.
9. Regional defaults now include all four requested hotspots, 2497, 4844, 2833
   and 1580. Native timestamp joins, region bands, quadrature weights, phase
   detection and scientific field definitions are unchanged.
10. The mapping launcher defaults now match the requested SKY partition/node,
    runs import/dependency/input preflight before its MPI stages, and records
    its failing command line number/exit status in `failure.txt`. A separate
    serial preflight can exercise every FE stage before a 32-core allocation.

The PETSc/mpi4py adapter remains in use for all Python-object collectives.
It preserves supplied communicator identity and subcommunicators, validates
rank/size and rejects null/out-of-range communicators. DOLFIN file I/O uses the
original communicator. Only the mixed validator's initialization-error reporter
may query world rank when DOLFIN itself failed before a mesh was constructed.

## Complete mapping execution chain

```text
run_phase_mapping_validation.slurm
  -> singularity exec --cleanenv IMAGE python preflight_diagnostics.py --static-only
       -> compile all mapping/shared modules to isolated .pyc files
       -> child-process direct and package imports
       -> exact standalone --help entry points
       -> child-process NumPy/DOLFIN/UFL/FFC/PETSc/mpi4py versions
       -> saved NPY geometry schema and submitted MyoSim layout checks
  -> for MPI size 1 and 32: singularity exec IMAGE mpiexec -np N python ...
       -> mpi_compat.py: real PETSc conversion/context and reversed-subgroup probe
       -> validate_ecm_quadrature_mpi.py: manufactured scalar/stress quadrature test
       -> validate_myosim_mixed_quadrature_ordering.py: production mesh/state packing
       -> export_verified_quadrature_geometry.py: geometry, vector ordering, weights
            -> shared production-mesh loader and cell/subspace map
            -> 32-rank saved ID-coordinate/wall matching
  -> scalar serial/MPI comparison
  -> geometry serial/MPI comparison
  -> seal_suite: validate all ten reports and input/mesh/source hashes
       -> mapping_manifest.json
```

`diagnostic_io.py` supplies exclusive report creation and shared root outcomes.
`mpi_compat.py` supplies communicator conversion and world API compatibility.
The mapping chain never runs `MyoFE.py` or imports cellular kinetics.

Regional analysis uses a separate Python >=3.6 environment with NumPy.
`run_phase_region_analysis.slurm` calls `analyze_phase_region.py`, which reads the
sealed mapping suite through the geometry exporter, streams CSVs using helpers
from `analyze_ecm_spatial_anomalies.py`, and uses the existing SVG renderer from
`diagnose_midwall_sarcomere_stdlib.py`. Optional displacement comparison uses
NumPy/h5py and its own explicit reference-coordinate lookup. Matplotlib is
optional for that tool. All imports match direct-script/package mode.

## Scope inspected

The mapping/region scripts above, both launchers, material-point mapping,
`compare_cycle_displacements.py`, `material_point.py`, the anomaly analyzer,
buffering/output comparisons, stride-one validator/storage estimator, and
their regression tests were reviewed. Read-only production-source review covered
`LV_simulation.py`, `mesh/mesh.py`, `dependencies/forms.py`,
`dependencies/fenics_2017_compat.py`, `dependencies/mechanical_health.py`,
`dependencies/buffered_output.py`, and MyoSim state construction. These production
files are unchanged. An AST/import inventory of all validation/postprocessing
Python files is delivered separately, distinguishing static scan coverage from
actual runtime validation. Ancillary material/fibrosis/inflation tools are not
called by the mapping or regional workflow and need their own FE workloads.

## Local verification and limits

The new tests exercise actual direct/package subprocess entry points, root
error propagation, exclusive reports, production state-vector dimensions,
explicit point IDs and XDMF time-window filtering. The real Bash mapping
launcher is executed against a marked synthetic Singularity boundary, covering
success, failure in the 32-rank scalar stage, partial results and refusing reuse.
Those shell tests validate orchestration; their synthetic reports do not establish
FE or MPI correctness. Separate tests check serial/32-rank command construction.

Local Python is 3.7.16/NumPy 1.16.5. DOLFIN and petsc4py are absent. mpi4py is
installed but cannot load an MPI DLL; Singularity and an MPI launcher are absent.
The actual preflight was run locally and correctly failed its runtime dependency
stage with a clear log and `passed: false`, leaving both FE rank counts unverified.
Real 1/32-rank gathers, FE compilation, HDF mesh loading, vector packing and
weight export must pass in the LCC container before those operations are claimed
validated. The container/image, LCC files and queue cannot be inspected here.

The broader repository test sweep also exposes three existing failures outside
this repair: two baseline-consistency checks compare `k_force=0.01` in the
baseline with `0.001` in validation configurations; one range-case test assumes
POSIX path separators on Windows. The test/config/generator files involved are
unchanged. Resolving the parameter discrepancy requires scientific review;
this repair does not change those values or suppress the failures.

Final local results are 76 passing focused diagnostic tests. The broad sweep
runs 180 tests, with 177 passing and the three failures listed above. All 15
validation Bash launchers parse individually. All validation/postprocessing/test
Python sources compile locally; the six mapping/shared entry modules also parse
with the legacy grammar. This grammar check supplements the emulated import
regression and does not establish execution under the container's interpreter.

## Update the LCC worktree

```bash
cd /home/sba431/MyoFE_H_diagnostics
git status --short
git pull --ff-only origin diagnostics/phase-regional-investigation
git rev-parse HEAD
export REPO=/home/sba431/MyoFE_H_diagnostics
export IMAGE=/home/sba431/fenics.img
export FAILED=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_mapping_6340522
export INPUT="$FAILED/instruction.json"
test -f "$INPUT"
```

`FAILED` is the previous launcher's default report path. If that job used a
custom OUT directory, use its actual archived instruction path. Derive the saved
output location from that instruction rather than assuming which simulation ran:

```bash
export DATA=$(singularity exec --cleanenv "$IMAGE" python - "$INPUT" "$REPO" <<'PY'
import json, os, sys
with open(sys.argv[1]) as handle:
    instruction = json.load(handle)
path = instruction['output_handler']['output_data_path'][0]
if not os.path.isabs(path):
    path = os.path.join(sys.argv[2], 'python_codes', path)
print(os.path.dirname(os.path.abspath(path)))
PY
)
test -d "$DATA"
export PREFLIGHT_ROOT=$(mktemp -d /mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_preflight_6340522_XXXXXX)
printf 'Input %s\nData %s\nPreflight %s\n' "$INPUT" "$DATA" "$PREFLIGHT_ROOT"
```

## Preflight before the next mapping job

Basic checks perform no FE solve. Run them on a host where site policy permits
container import checks, or inside a small allocation:

```bash
singularity exec --cleanenv "$IMAGE" python "$REPO/validation/preflight_diagnostics.py" \
  --repo "$REPO" --input-json "$INPUT" --saved-output "$DATA" \
  --output "$PREFLIGHT_ROOT/imports" --static-only
cat "$PREFLIGHT_ROOT/imports/preflight.json"
```

Then use one core to test the actual standalone FE/MPI entry points. This command
uses the same container-first MPI pattern as the mapping batch job:

```bash
PRE=$(sbatch --parsable --export=ALL --account=col_jfwe223_uksr \
  --partition=SKY32M192_D --nodelist=skylake047 --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G --time=00:30:00 \
  --output=phase_preflight_%j.out --error=phase_preflight_%j.err \
  --wrap='singularity exec --cleanenv "$IMAGE" python "$REPO/validation/preflight_diagnostics.py" --repo "$REPO" --input-json "$INPUT" --saved-output "$DATA" --output "$PREFLIGHT_ROOT/serial" --ranks 1')
PRE=${PRE%%;*}
squeue -j "$PRE"
```

After that job finishes, inspect `preflight.json` and stage logs:

```bash
sacct -j "$PRE" --format=JobID,State,ExitCode,Elapsed,NodeList
cat "$PREFLIGHT_ROOT/serial/preflight.json"
```

Require `passed: true` and `validated_ranks: [1]`. The report intentionally
leaves `mapping_workflow_validated` false until both rank counts have actually
run. That flag is limited to mapping execution; it does not certify actual-data
regional analysis or the mechanical interpretation of the hotspots.
Use a new child directory for retries; the preflight rejects existing output.
Missing dependencies, import errors, malformed inputs and failed child commands
include the failing stage and its log path.

For the separate analysis environment, point ANALYSIS_PYTHON to its executable
and run its diagnostic tests/CLIs on an analysis node:

```bash
cd "$REPO"
"$ANALYSIS_PYTHON" -c 'import sys,numpy,h5py; print(sys.version,numpy.__version__,h5py.__version__)'
"$ANALYSIS_PYTHON" -m unittest tests.test_phase_region tests.test_ecm_spatial_analysis -v
"$ANALYSIS_PYTHON" validation/analyze_phase_region.py --help
"$ANALYSIS_PYTHON" validation/compare_cycle_displacements.py --help
```

The preflight also accepts `--analysis-python PATH` if that interpreter is usable
inside the same container. It reports whether that environment was checked.

## Submit corrected 32-rank mapping validation

After reviewing the successful serial report:

```bash
cd "$REPO"
unset OUT
JOB=$(sbatch --parsable --export=ALL --dependency="afterok:$PRE" \
  --partition=SKY32M192_D --nodelist=skylake047 --nodes=1 --ntasks=32 \
  --cpus-per-task=1 --mem=32G --exclusive \
  validation/run_phase_mapping_validation.slurm)
JOB=${JOB%%;*}
export REPORTS=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_mapping_${JOB}
echo "$JOB $REPORTS"
squeue -j "$JOB"
```

The fresh SLURM ID creates a new directory. No prior job's report is overwritten.
If the preflight job has aged out of the scheduler, inspect its terminal status
and successful report before submitting without the dependency. If a baseline
is still running, retain the previous staged policy by adding its verified
`afterany` dependency. The mapping job runs diagnostic workloads only.

After completion:

```bash
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
cat "$REPORTS/preflight/preflight.json"
cat "$REPORTS/mpi_comm_1.json"
cat "$REPORTS/mpi_comm_32.json"
cat "$REPORTS/ecm_comparison.json"
cat "$REPORTS/geometry_comparison.json"
cat "$REPORTS/mapping_manifest.json"
tail -n 60 "phase_mapping_${JOB}.out"
tail -n 60 "phase_mapping_${JOB}.err"
```

The final manifest is created only after all ten mapping reports pass. A failed
job retains its partial reports and `failure.txt`, without a success manifest.
Keep the standalone preflight report and the final suite together for provenance.

## Scientific interpretation

Successful infrastructure and mapping tests establish only their tested mappings
and evaluations. Saved active stress still has the previously identified
post-reference-reset recording concern, and saved cardiovascular signals retain
their existing coupling stages. Exact hotspot deformation still requires original
CG2 displacement coefficients, reference fibers and matched material points.
These scientific limitations are included in analysis reports. No mechanical,
constitutive, kinetic, active-stress calculation or simulation parameter was
changed to obtain a diagnostic pass.
