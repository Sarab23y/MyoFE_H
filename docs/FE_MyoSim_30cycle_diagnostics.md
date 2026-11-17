# 30-cycle FE–MyoSim observer — implementation and validation status

> Historical prototype description. The follow-up implementation, corrected
> storage model and current validation handoff are documented in
> `FE_MyoSim_30cycle_followup.md`. Statements below about missing Newton hooks
> or unfinished event tooling describe the earlier prototype, not the new source.
> Neither version is certified for the full run without actual LCC validation.

**Prototype; NOT approved or ready for the 30-cycle run.** No SLURM job was
created/submitted, no simulation was rerun, and nothing was committed/pushed.
The dedicated full-run entry point deliberately refuses execution. The ordinary
production DOLFIN solve hides required trial/correction telemetry, and actual
FE/MPI/invariance tests have not been run in this Windows environment.

Base: `b5d051a14433c48257e1f6797863592dfd4d6889`. Isolated detached worktree:
`work/MyoFE_H_30cycle_diagnostics`. The original diagnostic checkout and master
were left alone. Prior FE-first/Phase-2 reports and the Phase-2 evidence ZIP were
read and reused. The ZIP's integrity was checked. Their mechanisms remain
conditional, not established historical causes.

## Changes and preserved calculations

Two existing execution files have tagged, disabled-by-default observation blocks:

- `python_codes/LV_simulation/LV_simulation.py`: construct the observer only for
  explicit `fe_cycle_diagnostic.enabled`; observe timestep start, immediately
  before/after the original solve, and after the original timestep; stop at a
  completed-cycle boundary; reject safety-ceiling exhaustion as incomplete.
- `python_codes/LV_simulation/half_sarcomere/half_sarcomere.py`: observe before and
  after original advection, before kinetics with its actual consumed calcium,
  after kinetics/before original feedback assignment, and after feedback.

Removing only the tagged blocks reproduces each original file exactly. A test
checks this against Git HEAD. No existing residual, Jacobian, constitutive law,
kinetics/advection/overlap/calcium implementation, circulation, baroreflex,
solver file, quadrature setting, tolerance or production launcher was edited.
Observation adds projections, copies and I/O; actual numerical equivalence still
requires the paired FE runs below. Source identity is not a substitute for that.

New files:

- `dependencies/fe_cycle_core.py` (under LV_simulation): pure cycle, kinematic,
  population, trigger and bounded event-window rules.
- `dependencies/fe_cycle_diagnostic.py`: runtime observer, mapping, stage arrays,
  owned-rank files, mixed-function HDF5 output and same-partition reopen checks.
- Three `demos/ECM/sim_input/base_instruction_Br_ECM_30cycles_FE_diagnostic*.json`
  templates: full target, 200-step enabled smoke and 200-step disabled control.
- `validation/prepare_fe_cycle_case.py`: archive the actual input, hash original
  mesh/image/source and dirty worktree, relocate only mesh/output paths, prepare
  exclusive new output directories, record a storage envelope.
- `validation/run_fe_cycle_case.py`: dedicated legacy-runtime entry point;
  cleanup/failure manifest; identical post-run mixed/vector capture for controls.
- `validation/run_fe_cycle_smoke.slurm`: configurable isolated worktree/image,
  1/32-rank smoke and checkpoint-reader invocations; no pinned production node.
- `validation/validate_fe_cycle_checkpoint.py`: independent DOLFIN mixed-space
  reconstruction; compare F/J/stretch/HSL using reference coordinates across
  partitions, not array order or coincidental IDs.
- `validation/compare_fe_cycle_smoke.py`: streaming CSV plus final owned-vector,
  population/calcium/feedback comparison; never certifies missing Newton checks.
- `postprocessing/report_fe_cycle_case.py`: global CSV, PV/J/stretch/residual
  overview, all-rank streamed events, bounded-frame F/J/HDF checks, XDMF parsing,
  scientific report with root cause explicitly unresolved.
- `tests/test_fe_cycle_diagnostic.py` and this document.

The directory prefix in the three templates is intentionally non-runnable;
prepare a frozen case first. No template points to old output directories.

## Cycle and timestep semantics

The real `heart_rate.implement_time_step` emits `new_beat` on activation's rising
edge after decreasing its counters. The first genuine marker starts an interval;
the 31st closes the 30th interval. Initial waiting time is excluded. Variable RR
periods are supported. The unit test executes that actual heart-rate function,
changes the quiescent period, and verifies all 30 start/end intervals.

Original mechanics/cellular update order is retained. The original CSV label is
the step's starting `t`, although the state has been advanced. The observer stores
both `legacy_time=t` and `accepted_time=t+0.001`. It does not relabel legacy CSVs.
The final marker step is fully accepted before stopping. It starts the following
activation at the boundary but does not integrate a partial additional cycle.

The original 60,001-iteration safety ceiling remains; it is not the termination
criterion. If 30 intervals do not finish before it, the case fails incomplete.
No baroreflex interval is extended beyond the original 5–60 s schedule. The
original ECM calibration at 40 s and its tolerance/reference-cycle requirements
are untouched. `calibration_time_reached` is distinct from successful calibration;
inspect the original ECM report for the latter. Thirty variable-rate cycles may
end pre-calibration. Smoke completion is never reported as 30-cycle completion.

## Implemented observation fields

At the accepted solver-return state, before the original hsl_old reset:

- All nine F components from `I+grad(u)` in the actual mixed CG2 field; determinant
  J; raw fiber stretch, HSL0/stretch-derived HSL and Green fiber strain; three
  singular values and their condition ratio; original fiber norms.
- Ground, collagen, Xi, ECM, active, pressure and total **PK2 fiber contractions**.
  The active coefficient is also stored separately; contraction multiplies it by
  `|f0|^4`, rather than silently assuming normalized fibers.
- Both pressure contractions `-p*J*f0.C^-1.f0` and the legacy
  `-p*f0.C^-1.f0`. These are new diagnostic fields, not replacements.
- Active equilibrium coefficient before reset; output-stage coefficient after
  the original timestep; their difference; hsl_old before and after reset.
- Independently assembled initial/final BC-applied residual L2, relative residual,
  solve wall time, net accepted mixed/displacement coefficient change. These are
  diagnostic mixed-unit norms, not a claim to expose the backend's iteration log.

Each point's five cellular stages retain actual y vectors, cellular HSL, raw
overlap calculated by the original function and stored overlap, cached stresses,
incoming stress feedback, delta HSL supplied to advection, calcium consumed by
kinetics, available scalar membrane states/parameters, and actual kinetic/force
parameters. SRX/DRX, bound/detached/myosin sums, conservation residual relative to
the initial sum, thin ON/OFF, min/max fractions, negative-bin counts and M_FG
(three-state) or M_PRE/M_POST (four-state) are derived from the recorded y.
Out-of-domain interpolation **query counts** follow the original bin positions;
this is not a measurement of pre-clipping interpolation values. Existing clipping
inside advection is untouched. Negative x-bin positions are not negative fractions.
Raw membrane y is copied alongside the cached membrane data and consumed calcium;
these can differ because the existing membrane data refresh occurs later. The
cellular HSL associated with overlap is saved separately from current FE HSL.

Global summaries record every step: J/stretch extrema, minimum singular value,
maximum stretch increment, inverted/low-J counts, negative overlap/population
point/bin counts, native pressures/volumes/flows, HR, activation, RR and available
baroreflex data. Configured physical hotspot coordinates and all face-neighbor
cells get every-step histories. IDs 2833/4844/1580/2497 are not assumed transferable.
Without supplied authoritative coordinates, hotspot tracking is explicitly absent.

## Mapping, event and checkpoint design

Runtime checks scalar owned IDs against the production list and verifies their
global uniqueness/coverage. Per-rank NPZ files store reference coordinates, scalar
IDs, cell IDs/connectivity/local quadrature index and reference fibers; mixed
local-to-global, owned ranges and per-cell parent/subspace maps are separate.
Global face neighbors are built from global connectivity. This does NOT replace
the existing scalar/mixed/component/scatter FE mapping suite. The manifest says
mixed and historical mapping remain unverified until the actual checks pass.

Triggers observe stretch<0.5, |delta stretch|>0.02, J<0.5/J<=0,
sigma_min<0.1, negative overlap/populations and nonfinite fields. These thresholds
do not constrain or repair mechanics. Ordinary solver errors are retained and
accepted pre-failure history is preserved. Large-Newton-correction/trial triggers
are not implemented because their required values are unavailable.

A 21-frame rolling buffer preserves 20 ms before a trigger; overlapping windows
are unioned and retain 50 ms afterward without duplicating event frames. Persistent
triggers retain every step and can be storage intensive. Selected complete fields
are emitted every 20 ms; mixed CG2 checkpoints every 100 ms plus event frames.
The initial accepted step is at 1 ms; no unmeasured 20-ms prehistory is invented.
Events near startup/failure/end can have truncated context. The completion manifest
records missing terminal post-event steps. Guaranteeing 50 ms after an event at
the last cycle boundary conflicts with stopping there; this remains a review issue.

HDF5 stores original mesh and mixed w, including CG2 edge DOFs, f0/s0/n0, HSL0,
hsl_old and y_vec. Copies are written, never assigned back to production Functions.
W contains CG2 vector, CG1, cavity Real, and five rigid-body Reals. Each emitted
mixed vector is reopened in the same partition and checked exactly; the separate
reader tests cross-partition reconstruction and quadrature geometry correspondence.
This is **not** a full coupled restart checkpoint. Pre-reset equilibrium hsl_old
also exists in the frame arrays; the HDF5 hsl_old is the original end-stage value.
Visualization XDMF is not used for reconstruction.
`reference_coefficients.h5` also archives resolved passive coefficient Functions
once; actual end-stage FE kinetic/force coefficient Functions accompany frames.
This preserves heterogeneous coefficients that a JSON default alone cannot prove.

Frame NPZ writes are exclusive, local I/O failures propagate collectively, root
metadata failures are shared, log errors surface at the next shared checkpoint,
HDF5 handles close in finally blocks and final JSON is atomically renamed. A hard
kill or backend/MPI hang cannot guarantee graceful output or close; absence of a
completion manifest means incomplete. Files must still pass the real smoke tests.
SIGTERM/SIGINT/SIGUSR1 handlers only set a flag; ranks check it after an accepted
step and follow the existing collective error/flush path. This is best-effort
boundary cancellation, not proof that a hard timeout/backend hang closes files.

## Validation status and unresolved requirements

Local focused tests cover real HR cycle counting, source-method observational
equivalence, exact tagged-block removal, unchanged scientific template values,
known F/J/stretch/singular values, populations, event union/bounds, explicit times,
exclusive writes, preparation safety, fake-backend checkpoint API behavior and a
synthetic HDF5/report fixture including a non-root event. **Fake-backend roundtrip
is not a successful DOLFIN checkpoint test.** See attached test logs for counts.

The broad suite has three failures reproduced in the untouched checkout: two
older fibrosis/growth fixtures disagree with their baseline k1 values (.001 versus
.01); one expects POSIX path separators on Windows. They were not corrected by
changing scientific parameters. Report these alongside the passing tests.

Not executed locally: actual LV smoke, 1/32-rank FE mapping, checkpoint FE readers,
same-initial-state physical equivalence, actual output finalization or performance.
No DOLFIN, cluster MPI/SLURM/container or LCC connection is available here.

**Implementation gaps:** the normal DOLFIN `solve(F==0,...)` hides trial iterates,
per-iteration corrections, linear solver statuses and iteration counts. Changing
to another Python Newton loop would violate the first-run comparability constraint.
No backend observer was invented or silently substituted. The full runner is
hard blocked until this design is resolved. The postprocessor currently provides
global/validation reporting and raw event/stage evidence, not the complete requested
phase-resolved stress/population plots, first-deviation/lag/causality analysis or a
validated stability diagnosis. Those are unfinished deliverables. No expensive
eigenvalue analysis, tangent modification or artificial sarcomere correction exists.

## Storage and resources

Preparation writes an uncompressed planning envelope assuming 5008 points, up to
128 population columns, five stages and 64 stage scalars, 64 derived scalars plus
F, a 4-MiB checkpoint allowance, and 60,001 safety-ceiling steps. A 50% disk margin
is included. These are assumptions to check against measured smoke frames, not
available-disk or memory validation. See storage_estimate.json/static_checks.json.
Persistent events can demand several TiB; no silent downsampling is performed.
Original CSV/log sizes and Python/backend object memory are additional. The bounded
array estimate alone is not total RSS. Measure actual frame bytes, MaxRSS, disk and
per-step timings before revising the envelope or choosing a full allocation.

The smoke launcher provisionally inherits the old 180-GiB/32-rank ceiling and asks
two hours; it does not claim measured requirements. Select a partition/node after
checking current jobs. No production node is pinned. Full-run memory/walltime cannot
be chosen honestly without sacct and the instrumented smoke. Thirty cycles at an
unchanging 0.923-s RR would span 27.69 s plus initial waiting; that is illustrative,
not a duration setting or a prediction under baroreflex.

See `LCC_Commands.md` in the delivery bundle for exact staged commands. **Do not
submit a full job now.** Local unit results cannot supply the missing approval and
LCC evidence; the runner refuses the full mode even with a fabricated readiness file.
