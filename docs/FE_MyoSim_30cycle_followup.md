> Compatibility update: the confirmed production image uses **DOLFIN 2017.1.0**. See [DOLFIN_2017_Compatibility.md](DOLFIN_2017_Compatibility.md); its strict image/header/symbol and runtime probes supersede the earlier 2019-only guard.

# Follow-up: native Newton observation and source-stage event analysis

**Readiness: not demonstrated. Do not submit the 30-cycle case.**
Continue the existing isolated worktree, based on b5d051a. No source commit, push,
PR or SLURM submission was made. The old 15-file delivery is retained separately.
Actual DOLFIN, container compilation, cluster MPI and paired LV tests remain
unexecuted in the Windows environment. The full entry point remains blocked.

## Native observation without replacing Newton

The ordinary high-level solve remains unchanged in nsolver.py. A new optional
Linux/x86-64 ELF library forwards `NewtonSolver::converged` and `update_solution`
to their original libdolfin implementations exactly once with the original
arguments. It never assembles F/J, updates x itself, changes parameters or
substitutes convergence acceptance. The callback records after the original
predicate and around the original update. If a configured backend returns
nonconvergence without throwing, that return is preserved and reported separately.

The design follows the versioned [DOLFIN Newton source](https://raw.githubusercontent.com/live-clones/dolfin/2019.1.0/dolfin/nls/NewtonSolver.cpp)
and [variational solver interface](https://raw.githubusercontent.com/live-clones/dolfin/2019.1.0/dolfin/fem/NonlinearVariationalSolver.h).
These are source references, **not validation of the container binary**. Actual
symbol export/preemption, C++ ABI, MPI ordering, GIL re-entry and numerical
invariance must be demonstrated by the provided probes. No fallback Python Newton
loop exists. Unsupported version/criterion, missing callbacks or callback errors
fail diagnostic validation. No DOLFIN library or production image is replaced.

Build inside the actual image with build_native_newton_observer.py. It discovers
exported symbols rather than assuming mangled names, compiles against the image's
headers/pkg-config, and archives compiler, source, symbol and library hashes.
Preload only in the diagnostic child processes; do not export LD_PRELOAD globally.

Recorded per solve: iteration zero and every residual evaluation; native residual
and relative residual; backend convergence; cumulative linear iterations; applied
displacement correction L2/max and actual relaxation; trial global J min/max and
minimum fiber stretch, with extreme-point owner/IDs; separate solver-return state
extrema. Displacement norms exclude pressure/Real DOFs. Raw correction norms are
derived from applied norms when relaxation is nonzero. No line search/damping is
introduced. Native errors do not unwind through a ctypes callback; ranks propagate
observable errors coherently and the original solve is allowed to return/throw.

Cardiac-cycle labels are obtained after the actual HR update. Zero denotes initial
waiting; boundary-marker states open the next interval. Trial values stay labeled
trial even if the native predicate returns true; accepted values are tagged only
after the original high-level solve returns. A nonconverged return is not relabeled
as convergence. Observed wall time explicitly includes callbacks; accepted-state
projection time is separate. Jacobian eigenvalues/condition numbers are not invented.

## Control arms and direct checkpoint validation

Prepare three identical-physics 200-step arms at each MPI size:

- `pure-control`: original disabled path, no native library or timestep observer;
  common read-only initial and final state capture only.
- `control`: minimal common read-only witness for per-step coefficients, HSL,
  stretch, calcium, cellular/FE populations, equilibrium active stress and native
  Newton counts/residuals/corrections. It is explicitly **not** a completely
  unobserved arm.
- `smoke`: full diagnostic observer and event recording.

compare_fe_cycle_smoke.py compares every witness/enabled native sample and the
pure arm's legacy histories/final coefficients. Initial vectors, spaces and cellular state must match exactly before trajectory comparison. Numeric physical tolerances are
atol=1e-10, rtol=1e-9; native residual/correction comparisons use atol=1e-12,
rtol=1e-9; iteration/convergence/identity fields must match exactly. Differences
raise an error rather than relaxing acceptance. These tests have not run on LCC.

smoke_cg2_checkpoint.py writes a manufactured quadratic field on the actual mesh
in the original mixed CG2/CG1/Real/5-Real space. It proves the fixture has nonzero
CG2 edge information, checks exact same-space coefficients after reopening, and
calls the independent new-mesh reader for F/J/stretch/HSL and physical-coordinate
correspondence. The reader checks element metadata and binds its report to the
checkpoint SHA-256. This is a manufactured field, not an LV trajectory or restart.

smoke_native_newton.py uses a separate manufactured vector reaction-diffusion
problem. The degree-two gradient term avoids a pure underintegrated CG2 mass
nullspace. This term exists only in that test, not the LV residual. It checks native
reachability, direct correction measurements, constant initial F=I and identical
solutions with the callback inactive/active. A deliberately failing callback on
one rank tests deferred error handling without changing the manufactured solution.
smoke_event_mpi.py tests the actual communicator's non-root/global event union
with explicitly artificial markers; it does not claim FE mapping validation.

## Event chronology and output validation

analyze_fe_event_causality.py is integrated into the enabled report. SQLite stores
joins/history on disk; memory holds one rank frame and a short plot window.
Overlapping point event windows are coalesced into episodes; every native sample
and every above-threshold jump remains in CSV. It checks explicit times and IDs,
F/J/stretch/reference consistency, and links the original population arrays and
CG2 state files. It never assigns an old integer ID to a new physical point.

The chronology includes the previous native FE state, consumed versus raw/cached
calcium, supplied HSL advection increment, population changes, incoming/cached
force feedback, initial/final equilibrium active coefficients, native Newton
trials, accepted deformation, reference reset/output stress, and next-step overlap.
The report separates trigger onset from the episode's largest jump and identifies
missing pre/post samples or stages. The +50-ms rule requires an exact native sample;
recovery/reversal/oscillation thresholds are disclosed. Ordinary kinetic changes
before a mechanical event are not labeled abnormal causes. Root cause remains
unresolved without independent mechanical/stability evidence.

Outputs: event_index.sqlite, event_timeline.csv, native_newton.csv,
native_jump_intervals.csv, global_newton_events.jsonl, event_summaries.json,
chronology_report.json, and native event/Newton plots. PK2 fiber contractions are
plotted together; scalar active coefficients remain explicitly separate in tables.
Global pressure/volume/flow context retains the original staggered update stages;
no exact valve event is inferred. Per-iteration pointwise stress is not claimed:
initial/final point stresses and global trial extrema are what is recorded.

check_fe_cycle_output.py checks native timestamps, configured CSV presence and
point headers, final partial batches, HDF signatures/readability, complete XML/HDF
references, hash-bound FE-reopen reports, cycle intervals, native status counts,
and every-rank event-frame unions. Startup/terminal context gaps remain explicit.
Geometry startup CSVs are one-row reference metadata, not 50-ms histories.
Signature creation alone cannot satisfy the independent FE reconstruction check.

The optional 37-step controlled-interruption smoke exercises the original
RuntimeError collective-flush path after a fully completed timestep, without
changing any physical equation or solver setting. It must report its exact
intentional reason and boundary; another failure cannot pass as that fixture.
Hard kills/MPI hangs cannot guarantee graceful finalization.

## Corrected findings and resources

The old failure explanation incorrectly called the mismatched input `k_1`.
Direct recursive comparison identifies **myofilaments/k_force**, .01 in the
generic base instruction versus .001 in five older growth/fibrosis fixtures.
Two existing equality tests fail for that reason in the untouched checkout. The
third expects POSIX path separators on Windows. No related model/fixture values
were changed; whether those old cases should match requires a separate review.

The old 49-GiB no-event estimate omitted routine stage arrays. Keeping the existing
profile unchanged, the 60,001-step planning cap has 601 checkpoints and 2400 other
20-ms field payloads: about **177 GiB** with the existing binary payload assumptions
and 50% margin. Persistent event-frame payload planning remains **3.73 TiB**.
Native JSON is a separate category: a deliberately conservative 50-iteration,
32-rank, 2-KiB-record assumption yields about **560 GiB** at that cap, with margin.
Original/baroreflex logs, postprocessing SQLite/CSV and Python/backend memory are
additional. These are partial planning envelopes, not measured total requirements.
The rolling array envelope remains .89 GiB globally, excluding object/backend RSS.

Actual runtime, field dimensions, compression, native iteration distribution,
MaxRSS and disk growth must be measured on LCC. Initial quiescent 200-step data
cannot establish later-event compression or runtime. Do not reduce frequencies,
fields, tolerances or parameters silently to fit a quota. Full resources remain
unselected. The smoke allocations are provisional ceilings, not measured needs.

Use the new delivery's LCC_Validation_Handoff.md for exact commands. Historical
source fidelity additionally requires comparing the actual production source or
archived source with this base; same input JSON alone cannot establish it.
