# Phase-resolved regional diagnostics: review and LCC runbook

Base: `4165d0eb6cbf7daa6403c435f7764df89101f3c4` on `master`.
Local development branch: `diagnostics/phase-regional-investigation`.
No production equations, kinetics, coupling, solver, simulation parameters,
or production launchers are changed. No push, PR, or LCC submission was made.

## Environment finding

`cascadeb024` and `cascadeb022` are SLURM compute-node names, **not containers**.
The production launcher selects `cascadeb024`; the existing scalar and mixed
validation launchers select `cascadeb022`. All name `/home/sba431/fenics.img`
and launch `singularity exec --cleanenv IMAGE mpiexec -np N python ...`.
No different image is established by these launcher files.

The new mapping launcher selects `cascadeb024` and the production image,
rank count (32), working directory convention and SCOTCH MPI partitioner.
It requests one exclusive node. For the first staged validation, submit with
an `afterany` dependency on the actual baseline job ID, so diagnostics wait for
that simulation to terminate. Current occupancy/completion is unverified here.
It records hostname, image/input SHA256, Git status/commit, Singularity,
Python, NumPy, DOLFIN and MPI versions. Runtime identity, image immutability
since production started, node availability and current simulation progress
cannot be verified from this Windows session. A checksum taken now cannot
prove the historical image checksum. Compare with production records if any.

The existing MyoSim mixed-space validator did not explicitly select the
SCOTCH partitioner used by production MeshClass. Its diagnostic setup now
does so for MPI, rather than relying on a site's default. This changes only
the validation test, not MeshClass or the simulation.

## Files and reuse

- `validation/validate_myosim_mixed_quadrature_ordering.py`: existing
  forward/inverse mixed-state packing validator; explicit SCOTCH selection
  and partitioner reporting, plus mesh/submitted-input checksums added.
- `validation/export_verified_quadrature_geometry.py`: reuses its mesh loader
  and authoritative cell/subspace map helper. Audits separate VectorElement
  point-major packing, exports scalar IDs, reference coordinates, containing
  cell vertices, local quadrature index, mesh wall-depth values and assembled
  reference-volume integration weights. Compares saved geometry by explicit
  ID and binds it with file hashes. Cross-rank comparison uses physical
  coordinates, weights and wall depth, not rank-dependent global IDs.
- `validation/run_phase_mapping_validation.slurm`: runs the existing
  manufactured scalar/direct-stress test and production-mesh mixed-state
  test on 1 and 32 ranks; runs the geometry audit on both and checks their
  coordinate correspondence. Only the 32-rank audit checks production IDs.
- `validation/analyze_phase_region.py`: reuses spatial path/header helpers
  from `analyze_ecm_spatial_anomalies.py` and the SVG renderer from
  `diagnose_midwall_sarcomere_stdlib.py`. Both reused files are unchanged.
- `validation/run_phase_region_analysis.slurm`: serial Python 3/NumPy job;
  requires an explicitly configured `ANALYSIS_PYTHON` executable and a sealed,
  passing mapping suite. Both launchers default to the diagnostic worktree and
  archived submitted input, and reject the production checkout path.
- `tests/test_phase_region.py`: small synthetic fixtures and end-to-end
  tests, requiring NumPy but no DOLFIN, SciPy, pandas or Matplotlib.

## Exact commands after approval and publication

These are future instructions, not jobs already submitted or a branch already
pushed. After approval, push the diagnostic branch to `Sarab23y/MyoFE_H`, fetch
only that branch, then create a separate worktree. No `pull`, checkout, reset or
patch application in the production working tree is required. Do not paste
credentials into chat.

Create an isolated diagnostic worktree. This leaves the existing production
checkout and its uncommitted files alone. If either destination or branch
already exists, inspect it instead of replacing it.

```bash
git -C /home/sba431/MyoFE_H status --short --branch
git -C /home/sba431/MyoFE_H rev-parse HEAD
git -C /home/sba431/MyoFE_H fetch origin \
  refs/heads/diagnostics/phase-regional-investigation:refs/remotes/origin/diagnostics/phase-regional-investigation
git -C /home/sba431/MyoFE_H worktree add \
  -b diagnostics/phase-regional-investigation \
  /home/sba431/MyoFE_H_diagnostics \
  refs/remotes/origin/diagnostics/phase-regional-investigation
cd /home/sba431/MyoFE_H_diagnostics
git diff --check
```

The bundle's patch is also available for review or approved offline application
to a separate worktree based on `4165d0eb6cbf7daa6403c435f7764df89101f3c4`.
If a diagnostic branch/worktree already exists, inspect it; never force-update,
remove or overwrite one that jobs may be using.

Before mapping validation, verify the run's actual input, mesh and producer
commit against the diagnostic baseline. The input copied by the production
launcher is preferable to a subsequently edited repository input:

```bash
export REPO=/home/sba431/MyoFE_H_diagnostics
export DATA=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1/sim_output
export INPUT="$(dirname "$DATA")/instruction.json"
export IMAGE=/home/sba431/fenics.img
test -f "$INPUT"
git -C /home/sba431/MyoFE_H log -5 --oneline
ls -l "$DATA/output_metadata.json" "$DATA/data.csv"
cat "$DATA/output_metadata.json"
```

If the copied input is unavailable, identify the actual submitted input
from the production log before using a repository input as a substitute.
The geometry loader resolves the configured mesh relative to the diagnostic
worktree's `python_codes` directory. The unchanged tracked baseline mesh must
exist there; a saved-coordinate mismatch blocks analysis rather than being
silently remapped. A different producer version needs review before results
are interpreted. No production trajectory can be inferred from master alone.

Inspect the queue and identify the actual baseline job from its name, node and
submitted command, rather than guessing a job ID:

```bash
squeue -u sba431 -o '%.18i %.24j %.10T %.20N'
read -r -p 'Actual baseline job ID: ' BASELINE_JOB_ID
scontrol show job "$BASELINE_JOB_ID"
```

If it is running or pending, submit with the dependency below. No simulation
is run by this diagnostic job:

```bash
cd "$REPO"
MAP_JOB=$(sbatch --parsable --dependency="afterany:${BASELINE_JOB_ID}" \
  validation/run_phase_mapping_validation.slurm)
MAP_JOB=${MAP_JOB%%;*}
echo "$MAP_JOB"
squeue -j "$MAP_JOB"
```

`afterany` allows diagnosis of stable partial output if the baseline exits
unsuccessfully; it does not establish successful 60-second completion. If an
already-terminated job has aged out of the scheduler, verify its terminal state
with `sacct` before submitting without the dependency. Do not cancel, requeue,
pause or otherwise change the baseline job.

After mapping validation finishes:

```bash
export MAPPING=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_mapping_${MAP_JOB}
cat "$MAPPING/environment.txt"
cat "$MAPPING/ecm_comparison.json"
cat "$MAPPING/geometry_comparison.json"
cat "$MAPPING/mapping_manifest.json"
```

Inspect `mixed_1.json`, `mixed_32.json` and `geometry_32.json` as well. Every
report must pass. Exceptions, missing mesh wall-depth data, packing defects,
duplicate coordinates, invalid weights, or saved-coordinate mismatches are
failures to investigate, not reasons to relax tolerances.

Optional node comparison after reviewing site/node availability, retaining the
same image, input and baseline dependency:

```bash
sbatch --nodelist=cascadeb022 --dependency="afterany:${BASELINE_JOB_ID}" \
  validation/run_phase_mapping_validation.slurm
```

Compare both jobs' `environment.txt` and reports. This overrides only the new
diagnostic launcher's node request. It does not edit the production launcher.
`cascadeb022` availability is unverified, so delayed submission on `cascadeb024`
is recommended first. Slurm documents `afterany` as waiting for termination;
`--exclusive` normally prevents sharing, subject to partition policy:
[official sbatch documentation](https://slurm.schedmd.com/sbatch.html).

Analysis requires a stable dataset. Prefer completed, flushed output, or a
filesystem-supported immutable snapshot. Do not copy hundreds of GB without
checking space. Differing flushed row counts are handled by selecting cycles
whose timestamps are present in every analyzed field. Partial CSV rows,
off-grid timestamps, changed file sizes/mtimes during processing and fewer
than five complete consecutive cycles cause explicit failure. If analysis
fails, its output directory is not a valid completed report; use a new output
name for retries. No production output is overwritten or repaired.

Verify Python 3/NumPy independently of the legacy FEniCS container:

```bash
read -r -p 'Absolute path to your Python 3 executable with NumPy: ' ANALYSIS_PYTHON
export ANALYSIS_PYTHON
test -x "$ANALYSIS_PYTHON"
"$ANALYSIS_PYTHON" -c 'import sys,numpy; print(sys.version); print(numpy.__version__)'
cd "$REPO"
"$ANALYSIS_PYTHON" -m unittest tests.test_phase_region tests.test_ecm_spatial_analysis tests.test_buffered_output -v
```

The launcher has no default `python3` fallback. Set `ANALYSIS_PYTHON` to an existing approved
Python >=3.6 environment with NumPy. Do not use the legacy Python 2 FEniCS
interpreter for post-processing, and do not modify that container to install
analysis packages.

Review the mesh in ParaView to confirm the apex-to-base axis and that the
mesh field `ellipsoidal/endo_dist` represents normalized wall depth. Numerical
agreement with saved `norm_dist_endo.npy` alone cannot establish anatomical
meaning. Then, **if Z is confirmed**:

```bash
export LONG_AXIS=2
export WALL_DEPTH_VERIFIED=1
unset START END
cd "$REPO"
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterok:${MAP_JOB}" \
  validation/run_phase_region_analysis.slurm)
ANALYSIS_JOB=${ANALYSIS_JOB%%;*}
echo "$ANALYSIS_JOB"
squeue -j "$ANALYSIS_JOB"
```

Or run directly on an allocated analysis node:

```bash
"$ANALYSIS_PYTHON" "$REPO/validation/analyze_phase_region.py" \
  --data "$DATA" --mapping "$MAPPING" --input-json "$INPUT" \
  --axis "$LONG_AXIS" --cycles 5 \
  --output /mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_region_review_001
```

With `START` unset, selection uses the latest five consecutive complete cycles
available in every analyzed field, including stable partial output ending before
40 seconds. Reports include actual end time, selected interval and whether the
selection starts before calibration. Use `START=40` only for an explicitly
requested post-calibration analysis with enough data. An explicit start never
falls back to another window or fewer cycles.

The CLI itself requires all eight reports plus `mapping_manifest.json`. Report
and diagnostic-source hashes, rank counts, every mixed-test rank, mesh/input
identity, SCOTCH selection, vector mapping and weights must pass. Saved geometry
arrays are hash-bound and their ID-coordinate tuples independently rechecked.
Cell vertices/local quadrature indices are required; points must lie inside
their reference cells. Positive finite weights must sum to reference volume.
Stale, incomplete or inconsistent evidence blocks staged analysis.

## Interpretation and expected outputs

The analyzer prefers forward aortic/inlet flow for valve-event estimates;
it can fall back to ventricular/aortic or atrial/ventricular pressure
differences, or explicitly identified model venous-inlet/ventricular pressure,
then volume derivatives. The current circulation exposes venous
inlet flow, not an independent atrial chamber measurement. Venous pressure
is never relabeled as atrial pressure. `new_beat` can estimate the cycle
boundary if inlet closures cannot be found; this is explicitly labeled as an
MVC estimate. Cycles with unresolved phase ordering, valve chatter/reopening,
overlap or global time gaps are rejected. No fixed cardiac phase times are used.

`report.json` records event times, ED/ES at estimated inlet/aortic closure,
peak pressure, volumes, phase durations, uncertainty of at least one sample
interval, rejected cycles, coverage, geometry hash and anomaly screens.
These are sampled forward-flow events, not measured valve leaflets. Thresholds
are in original saved flow/volume units and recorded in the report parameters.
Source review found that saved flow is cached from circulation ODE evaluations,
while ventricular pressure is updated after mechanics. The additional staging
uncertainty is unquantified (`coupling_stagger_uncertainty_s: null`), separate
from sample spacing. `exact_valve_events_identified` remains false. Reports flag
reverse-flow samples, pressure/flow sign disagreements, and volume changes in
nominal isovolumetric intervals rather than forcing consistency or relabeling
saved states.

`region_geometry.csv` contains **all** points in inclusive wall-depth and
long-axis fraction bands [0.3,0.7]. It does not target an expected count of 304
and does not reduce the selection to eight points. Near-duplicate reference
coordinates cause failure for review. Reference weights are assembled using
the production scalar quadrature basis against reference `dx`, checked positive
and against total mesh reference volume. They are not deformed-volume weights.

For each core field (`hs_length`, `fiber_stretch_raw`, `fiber_strain`,
`active_stress`, `Sff_mesh`, `ecm_instantaneous_fiber_stress`):

- `*_region.csv`: every aligned interior sample, point mean/std, min/max and
  their IDs, median, p05/p25/p75/p95, and separately named reference-volume
  mean/std when verified weights exist. Percentiles remain unweighted point
  percentiles. Shortening fractions mean negative stepwise HSL/stretch change;
  stretch below one is reported separately. Xi-zero fractions use exact saved
  zero, without a hidden tolerance. Both point and volume fractions are labeled.
  Raw stretch additionally reports region/whole-mesh screen fractions and
  whole-mesh extrema with their point IDs, to distinguish local from broad events.
- `*_hotspots.csv`: exact saved values for IDs 2833, 2497, 4844.
- `*_cycleN_PHASE_points.csv`: regional points and hotspots, phase extrema,
  first/last interior values, signed change and maximum drop from first sample.
- `phase_summary.csv`, `cycle_variability.json`, `point_cycle_variability.csv`:
  regional and per-point phase comparisons across all five cycles, including
  consecutive differences, ranges and between-cycle standard deviations.
- `regional_overview.svg`, `*_hotspots.svg`, `*_phase_cycles.svg`,
  `cardiovascular.svg`, `pressure_volume.svg`: standalone browser-viewable plots.
  Phase overlays normalize each independently detected phase duration to [0,1]
  for plotting only. CSV values retain their original times and resolution.

Available cytosolic/SR calcium and release/uptake fields are analyzed as optional
fields; missing ones are listed. Missing core stress/mechanics fields cause
failure. No total-passive field is substituted for Xi. Historical total-passive
and total-stress diagnostics are excluded here because their per-point producer
mapping needs separate validation.

Memory scales with global scalar timestamps, mesh size, region size and the
requested cycle count, not with the full space-by-time history. CSVs are read
twice: a timestamp/header coverage pass, then one field at a time. Spatial
statistics and hotspot rows are written immediately. Plot histories are bounded;
no numerical output is downsampled. Selected phases use left-closed/right-open
intervals. Phase changes use interior samples, not interpolated valve states.
Temporal summary means are explicitly sample means; they are not time integrals
for irregular time spacing.

## Scientific limits

The saved active-stress recording concern is unchanged. Calcium/stress staging
is unchanged. These tools do not prove that saved fields equal the equilibrium
stress or that a deformation is physical. Extreme-stretch screens (default
below 0.5 or above 1.35) are configurable investigation thresholds, not healthy
reference limits or constitutive constraints. Events are capped at 1,000 entries
in JSON; full regional extrema and hotspot CSVs remain available.

Original FE displacement, full F/J, pressure and neighboring-element checks at
the reported events remain necessary. Global IDs are only accepted when the
same production-rank reconstructed space matches saved coordinates. If that
fails, do not attach hotspot IDs to a newly partitioned mesh by array index.
No negative-population clipping or other model correction is introduced.

Local validation uses ParaView's Python 3.10.11/NumPy 1.25.2. LCC/DOLFIN mapping
tests and actual-data phase results remain unexecuted in this session.
Final-review tests cover the sealed-suite gate, stale/missing reports, wrong
ranks, input mismatches, reference-volume sums, cell containment, default
partial-data window selection and the explicit phase-uncertainty flags.
