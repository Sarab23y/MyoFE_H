# Job 6340763: Bash repair and isolated HSL–MyoSim validation

Diagnostic-only changes on `diagnostics/phase-regional-investigation`.
No production mechanics, constitutive law, kinetics, coupling, parameters,
launchers or simulation result files are changed. No production simulation
is launched by the commands below.

## Evidence classes

- **Source-confirmed:** execution order or expressions in inspected source.
- **Fixture-demonstrated:** actual source functions executed with explicitly
  synthetic initial populations/feedback. This is not historical replay.
- **Historically observed:** saved scalar HSL values/increments supplied in
  the preceding investigation. Actual material identity is still provisional.
- **Unresolved:** requires target-runtime, original-state or full-order FE data.

## Bash failure and repair

The reported failure is consistent with older Bash nounset behavior: an
initialized but empty array can be treated as unset when `${EXTRA[@]}` is
expanded under `set -u`. Modern Bash accepts it. A successful local modern-Bash
test therefore cannot refute the LCC failure.

Two diagnostic launchers had this risk:

1. `validation/run_phase_mapping_validation.slurm`: `EXTRA=()` for the serial
   geometry call, populated only for the 32-rank call.
2. `validation/run_phase_region_analysis.slurm`: `WINDOW=()` when START/END
   are unspecified.

Both now use `set --` and `"$@"`, preserving each argument and optional pair
without expanding an empty array. `tests/test_bash_nounset.py` executes the
actual argument-building blocks for serial/parallel mapping and all four
START/END combinations, including paths with spaces. It syntax-checks every
`.slurm`/`.sh` under `validation/` and rejects remaining array expansions.
The whole-repository shell search found no other `[@]` expansions. Production
launchers were inspected by search only and were not modified.

Local Bash is **5.2.26 (MSYS)**. The original empty-array probe succeeds there;
that is logged, not presented as reproduction of old Bash. No Bash-version
compatibility environment variable is used to pretend this is old Bash.
The regression supports `DIAGNOSTIC_TEST_BASH=/bin/bash` and runs automatically
inside the mapping launcher's environment capture on LCC. Thus the actual
site version and original failure probe will appear in `environment.txt`.
The complete mocked-container launcher test also supports the selected Bash.
It is explicitly a control-flow fixture, not a successful FE test.

**The LCC Bash version and its regression result remain unresolved until that
job runs.** A separately configured stdlib Python 3 can be selected with
`BASH_TEST_PYTHON`; it does not need NumPy for the Bash test.

## Mapping audit and current status

The scalar validator previously used a UnitCube mesh. Its existing deterministic
fields can now be evaluated on the actual production mesh by `--input-json`.
The mapping launcher and preflight workflow supply that option. The default
manufactured-mesh CLI remains available for older standalone workflows. The
report explicitly distinguishes manufactured fields from historical deformation
and records input/mesh hashes. The suite seal now requires the same mesh and
submitted input for scalar, mixed and geometry reports.

The geometry exporter now additionally:

- Records owner rank and owned local index at every actual scalar point.
- Checks actual DOLFIN ownership ranges and local-to-global indices.
- Compares production `dofmap.dofs()` with authoritative local/global IDs.
- Inserts unique markers into the actual scalar FE vector, reads them back,
  gathers rank payloads, scatters by production IDs, and verifies every global
  index/value with no holes/duplicates.
- Reports `owned_index_and_rank_scatter_verified` and full scatter verdict.
  The suite seal refuses reports without this new evidence.

The pure-Python scatter helper has corruption tests for reordered ranks, wrong
IDs, duplicate IDs, bad owned indices/ranges, wrong markers and missing values.
These tests exercise fixtures; only a real FE invocation validates DOLFIN.

| Check | 1 rank | 32 ranks |
|---|---|---|
| Actual communicator and MPI size | NOT RUN | NOT RUN |
| Scalar Quadrature on submitted production mesh | NOT RUN | NOT RUN |
| MyoSim mixed component packing/unpacking | NOT RUN | NOT RUN |
| Scalar/vector geometric point correspondence | NOT RUN | NOT RUN |
| Owned indices, ownership ranges, rank/global scatter | NOT RUN | NOT RUN |
| Saved production ID-coordinate-wall tuples | Not comparable by serial ID | NOT RUN |
| Cross-rank physical-coordinate comparisons | NOT RUN | NOT RUN |

This Windows session has no configured LCC SSH target/credentials, no local
DOLFIN, `sbatch`, Singularity or `mpiexec`. No LCC job was submitted here.
Infrastructure fixes and Python fixture tests do not establish FE mapping.

## Exact LCC commands

Use the existing diagnostic checkout only. Inspect any local changes before
pulling; do not reset or stash automatically:

```bash
cd /home/sba431/MyoFE_H_diagnostics
git status --short --branch
git pull --ff-only origin diagnostics/phase-regional-investigation
git rev-parse HEAD
export REPO=/home/sba431/MyoFE_H_diagnostics
export IMAGE=/home/sba431/fenics.img

# Reuse the actual submitted instruction and saved-output directory from
# job 6340763, not a repository configuration that may have changed.
: "${INPUT:?Set INPUT to the actual submitted instruction used by job 6340763}"
: "${DATA:?Set DATA to the saved-output directory used by job 6340763}"
export INPUT DATA
test -f "$INPUT"
test -d "$DATA"
test -f "$IMAGE"

# Optional: point this at an existing separate Python 3 if python3 is unavailable.
export BASH_TEST_PYTHON=${BASH_TEST_PYTHON:-python3}
/bin/bash --version
DIAGNOSTIC_TEST_BASH=/bin/bash "$BASH_TEST_PYTHON" -m unittest tests.test_bash_nounset -v
```

`INPUT` and `DATA` are deliberately required instead of guessing their values.
Their full paths were not provided to this session. Use the frozen instruction
and saved-output paths recorded for the failed job.

Submit one fresh exclusive diagnostic allocation on the previously used
`cascade001`/`CAL48M192_D`, separate from the production `cascadeb024` node.
The launcher runs 1-rank checks followed by 32-rank checks inside the same
allocation, with container-first MPI:

```bash
cd "$REPO"
squeue -w cascade001
unset OUT
unset DIAGNOSTIC_TEST_BASH   # job tests the site's /bin/bash
JOB=$(sbatch --parsable --export=ALL \
  --partition=CAL48M192_D --nodelist=cascade001 --nodes=1 --ntasks=32 \
  --cpus-per-task=1 --mem=32G --exclusive \
  validation/run_phase_mapping_validation.slurm)
JOB=${JOB%%;*}
export REPORTS=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_mapping_${JOB}
echo "Job=$JOB Reports=$REPORTS"
squeue -j "$JOB"
tail -n 40 "phase_mapping_${JOB}.out" "phase_mapping_${JOB}.err"
```

SLURM may queue an exclusive-node request; do not kill or move another job.
The command overrides the diagnostic launcher's SKY defaults. Each job creates
a new job-ID directory and refuses an existing directory. Previous failed
reports, production source and active output are untouched. Do not supply an
OUT below the simulation directory.

After completion, print an explicit per-report verdict; missing reports are
NOT RUN, not PASS. A failed early stage leaves later checks unexecuted:

```bash
cat "$REPORTS/environment.txt"
test ! -f "$REPORTS/failure.txt" || cat "$REPORTS/failure.txt"
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
"$BASH_TEST_PYTHON" - "$REPORTS" <<'PY'
import json, os, sys
root = sys.argv[1]
names = ['mpi_comm_1','mpi_comm_32','ecm_1','ecm_32','mixed_1','mixed_32',
         'geometry_1','geometry_32','ecm_comparison','geometry_comparison']
for name in names:
    path = os.path.join(root, name+'.json')
    if not os.path.isfile(path):
        print(name, 'NOT RUN / report missing')
        continue
    try:
        with open(path) as handle: r = json.load(handle)
    except Exception as error:
        print(name, 'FAIL / unreadable report', error)
        continue
    print(name, 'PASS' if r.get('passed') is True else 'FAIL')
    if name.startswith('geometry_') and name[-1].isdigit():
        for field in ['weights_verified','vector_point_major_ordering_verified_all_ranks',
                      'owned_index_and_rank_scatter_verified']:
            print(' ', field, 'PASS' if r.get(field) is True else 'FAIL')
    if name == 'geometry_32':
        print('  saved_mapping_verified', 'PASS' if r.get('saved_mapping_verified') is True else 'FAIL')
    for rank in r.get('ranks', []):
        print('  rank', rank['rank'], 'PASS' if rank.get('passed') is True else 'FAIL')
PY
cat "$REPORTS/mapping_manifest.json"
```

Require all ten reports and the sealed manifest. Inspect scalar mesh/input
hashes, geometry scatter details, mixed rank reports and reference-coordinate
comparisons. A missing manifest is incomplete validation. Passing this suite
does not prove the historical CG2 deformation or authenticate a different
producer environment.

## Isolated source-function HSL–MyoSim experiment

`validation/validate_hsl_myosim_coupling.py` imports actual cellular methods
without DOLFIN and executes the actual FE `MeshClass.return_cb_stress` method
body with numeric coefficient stand-ins. It performs **no FE equilibrium
solve**, no model correction and no historical replay. The input configuration
supplies parameters; its checksum is recorded, not treated as proof it produced
the saved run. All seven production source files are fingerprinted.

`validation/fixtures/hsl_coupling_events.json` labels observed HSL/increment
scalars separately from synthetic states. DOF 2833 is first, with the requested
increment **-59.0309348 nm**. Additional increments are +47.8825043 (2497),
+48.0651900 (4844), and +53.6436849 nm (1580). Scalar values do not establish
historical population identity.

The synthetic initial populations are SRX=.4, detached=.55, uniform bound
total=.05, thin on=.1, thin off=.9. Calcium starts at 1e-7, with the remaining
configured content in SR. Stored feedback is CB=100/passive=20 Pa; incoming
feedback is CB=300/passive=40 Pa. These are numerical fixtures, **not measured
historical populations, calcium inputs or stress feedback**.

The script records full y arrays and pool totals before advection, after actual
advection and after actual kinetics; positivity, head/thin mass, detached/bound
fractions, overlap, HSL, consumed calcium versus advanced membrane y, stored
versus incoming stress, and fixed-y numeric pre/post-reset stress. It then calls
the actual `update_data` to verify calcium-dictionary publication. The experiment
represents the next MyoSim catch-up call using the observed FE increment, not
the preceding FE solve that generated it.

Run in a separate Python >=3.7 with NumPy, SciPy and pandas. Do not assume the
site's default Python or the FEniCS interpreter has these dependencies:

```bash
cd /home/sba431/MyoFE_H_diagnostics
: "${COUPLING_PYTHON:?Set COUPLING_PYTHON to an existing Python 3.7+ with NumPy, SciPy and pandas}"
"$COUPLING_PYTHON" -c 'import sys,numpy,scipy,pandas; assert sys.version_info >= (3,7); print(sys.version,numpy.__version__,scipy.__version__,pandas.__version__)'
COUPLING_ROOT=$(mktemp -d /mnt/gpfs2_4m/scratch/sba431/demos/ECM/coupling_diag_XXXXXX)
"$COUPLING_PYTHON" validation/validate_hsl_myosim_coupling.py \
  --input-json "$INPUT" --output "$COUPLING_ROOT/source_fixture"
cat "$COUPLING_ROOT/source_fixture/coupling_report.json"
cat "$COUPLING_ROOT/source_fixture/population_stages.csv"
```

The tool refuses existing outputs and paths under `sim_output`/`mesh_output`.
For local reproducibility when archived INPUT is unavailable, the repository
candidate `demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json`
can be supplied, explicitly labeled as a candidate. That was the configuration
used for this session's fixture execution. Permanent unit tests verify the
stage contracts and output protection. They do not assert historical mechanics.

## Findings and confidence

| Finding | Evidence class | Meaning |
|---|---|---|
| Four event HSL increments | Historically observed scalar data | Input probes only; full historical state absent. |
| All four shifts move existing synthetic bound bins outside -10..10-nm support | Fixture-demonstrated with actual advection | Bound population becomes zero before kinetics; head mass remains one. Not proof of historical distribution loss. |
| Kinetics consumes old dictionary calcium while membrane y has advanced | Source-confirmed and fixture-demonstrated | Consumed 1e-7 versus advanced approximately 9.9814072e-8 in this activation-zero fixture. |
| Kinetics consumes stored total stress, incoming stress assigned afterward | Source-confirmed and fixture-demonstrated | 120 Pa consumed; 340 Pa assigned after kinetics. Neither is historical feedback. |
| Resetting the HSL increment changes fixed-y numeric active stress | Source-confirmed and fixture-demonstrated | Demonstrates expression dependence, not a real UFL re-evaluation at historical displacement. |
| Extreme HSL 198.815923 nm gives negative overlap | Fixture-demonstrated with supplied parameters | Actual overlap function returns about -0.144468; this is outside normal overlap geometry, not proof of physical collapse. |
| Sparse separate advection fixture produces detached fraction -0.0188148314 | Fixture-demonstrated | Nonnegative normalized initial populations; head mass conserved but positivity violated. Not any historical point state. |
| Original CG2 shortening, element inversion or equilibrium branch switch | Unresolved | Requires independent full-order FE data and validated identities. |

For fixed synthetic y, the actual FE stress method with the 2833 increment
produced -83,888.36253 Pa before the numeric reset and +17,940 Pa afterward,
without negative initial populations. Other fixture pre-reset values were
100,537.31993 Pa (2497), 100,852.45280 Pa (4844) and 110,475.35641 Pa (1580),
all with the same 17,940-Pa post-reset value. These are synthetic-state outputs,
not measured hotspot stresses. Negative stress alone does not prove a population
defect. Conversely conserved head mass does not guarantee positivity.

The source locations are `half_sarcomere/half_sarcomere.py:29–54` (ordering),
`membranes/membranes.py:41–49,72–79,103–104` (calcium publication),
`myofilaments/move.py:5–64` (quadratic advection/clipping),
`myofilaments/myofilaments.py:143–169` (overlap), and
`mesh/mesh.py:1303–1334` (FE active scalar), relative to
`python_codes/LV_simulation/`. The production reset/recording order is
`LV_simulation.py:1402–1417,2933–2967`: CB stress is sampled, hsl_old reset,
then the live active expression is evaluated for output. This source evidence
does not mean the preceding equilibrium used the post-reset stress.

## What would prove or refute a physical FE shortening event?

1. Complete the real 1/32-rank mesh/component/geometry/scatter suite and bind it
   to saved ID-coordinate tuples, actual input, mesh checksum and producer.
2. Recover original full CG2 displacement coefficients, mesh topology, FE DOF
   map/order, reference fiber Function/component mapping and hsl0 at both sides
   of an event. Also recover hsl_old before reset and full population state.
3. At the exact quadrature locations reconstruct `F=I+grad(u)`, `J=det(F)`,
   singular values and `lambda=||F*f0||`; verify unit reference fiber norms and
   compare HSL/reference ratios independently. Examine containing cells and
   face neighbors, not nearest global ID numbers.
4. Evaluate the **pre-reset** mechanical residual, active/passive/pressure
   stresses and Newton correction/residual history with the recorded state.
   Small residual alone does not prove stable/unique or well-conditioned
   equilibrium.

Vertex-only tetrahedron4 XDMF cannot reconstruct missing CG2 edge modes.
HSL/stretch/strain agreement is shared-expression consistency, not this
independent validation. If checkpoints exist, prefer read-only reconstruction.
If full coupled restart state exists, propose a targeted unchanged-physics
replay around one event only after approval. No complete simulation is needed
or authorized here, and no replay was executed in this task.

## Local validation

The deliverable bundle includes actual local test output, numeric fixtures,
full population-stage reports and four SVG plots labeled SYNTHETIC. Local
results must not be substituted for the NOT RUN LCC table above. The previous
investigation's broader historical analysis remains separate.

Final local result: **92 tests passed** (Bash/control-flow, MPI compatibility,
HDF5 paths, mesh/report binding, scatter corruption, source coupling, buffered
output, phase/regional analysis and mechanical/material-point diagnostics).
Python compilation and Git whitespace checks passed. The actual FE tests,
historical-population replay and the site's Bash regression are still NOT RUN.
