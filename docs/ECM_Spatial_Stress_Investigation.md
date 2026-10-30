# ECM spatial-stress investigation

## Evidence available in this checkout

The case configuration and launchers are present under `demos/ECM`, but the
reported job-6292554 JSON, logs, CSV summaries, and hotspot PNGs are not present
in this checkout. Conclusions that depend on their pixel/data contents must
therefore be reproduced from the LCC output rather than inferred.

## Source findings

* The ECM tensor is the sum of the already fraction-weighted ground-matrix and
  collagen PK2 tensors. Xi, active, and constraint-pressure stresses are not in
  that sum.
* The diagnostic formerly projected the ECM scalar into DG1 and then
  interpolated DG1 back to Quadrature. That is a smoothing/reconstruction and
  is not pointwise-equivalent at the material integration points. The
  diagnostic now projects directly into its degree-two Quadrature space.
* Spatial CSV columns are assembled with global scalar Quadrature DOF IDs. The
  output diagnostics reject duplicate/missing IDs and compare saved `lx/ly/lz`
  against tabulated quadrature coordinates. A downloaded subset is trustworthy
  only if it retains those headers and coordinate checks.
* `total_stress_spatial` was allocated but its calculation and assignment were
  commented out. New output now populates it as `f0 . S_total . f0`; this does
  not imply old all-zero files represented zero mechanical stress.
* Historical `bulk_passive` and `total_passive` CSV values were taken directly
  from DG1 vectors but labeled with scalar-Quadrature global DOF columns. Equal
  vector lengths do not establish equal basis ordering, so cross-field hotspot
  plots from those old files cannot be treated as same-material-point evidence.
  Both outputs are now evaluated directly in the scalar Quadrature space before
  global-DOF assembly.
* `Ell`, `Err`, and `Ecc` are cycle-strain snapshots, not continuously
  evaluated instantaneous strains. They are evaluated at the hard-coded
  systolic sampling instant relative to the corresponding stored diastolic
  deformation and are then held until the next snapshot. This sample-and-hold
  change does not turn them into instantaneous strains; analyses must not use
  repeated held values as independent strain measurements.
* Explicit `MPI.Finalize()` ran before Python/FEniCS interpreter teardown. It
  was removed so mpi4py performs orderly automatic finalization after FEniCS,
  PETSc, HDF5, and XDMF objects are destroyed.

## Offline analysis

The analyzer streams one spatial CSV at a time and does not require DOLFIN:

```bash
python validation/analyze_ecm_spatial_anomalies.py \
  /mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_provisional \
  --dofs 1757 1773 2317 2496
```

Windows PowerShell example:

```powershell
python validation\analyze_ecm_spatial_anomalies.py `
  "D:\ECM_40s_provisional" --dofs 1757 1773 2317 2496
```

It writes `analysis_ecm/ecm_spatial_anomaly_report.json`, a long-form selected
DOF history CSV, and plots when Matplotlib is available. Missing ECM-only files
remain explicitly missing; the utility never substitutes `Sff`, bulk stress,
or total stress. Exact physical FE L2 reproduction still requires the mesh and
quadrature weights or a saved calibration snapshot from a FEniCS run.

For a future run, exact ECM quadrature histories can be requested selectively
with an `output_handler.ecm_spatial_csv_fields` list. Leaving this option absent
does not disable any ECM calculation and avoids allocating several additional
full-run history tables.

The exact opt-in syntax is:

```json
"ecm_spatial_csv_fields": [
  "ecm_instantaneous_fiber_stress",
  "ecm_cycle_averaged_fiber_stress"
]
```

Only request fields needed for the test. At 8,001 saved states, each additional
double-precision history requires approximately `8001 * Nqp * 8` bytes before
Pandas/CSV overhead, so enabling all ECM fields can still create multi-GB
output. This option only controls recording; `record_time_step()` and the ECM
calculation remain active. Numeric CSV column names are the global scalar
Quadrature DOF IDs. `quadrature_global_dof.npy` stores those IDs explicitly and
`quadrature_dof.npy` stores the corresponding reference coordinates in the
same row order.

## FEniCS 2019.1 runtime validation gate

The direct material-point expression is projected into the degree-two scalar
`Quadrature` space with the legacy FFC `quadrature` representation, degree two,
and the `default` rule. It must not be declared production-validated solely
from source inspection. Run a short, non-remodeling MPI comparison in the
cluster image before another 80-second case:

1. At one saved state, evaluate the same ECM PK2 fiber expression by (a) the
   historical DG1-project/interpolate path and (b) direct Quadrature projection.
2. Save global DOF ID, reference X/Y/Z, old value, new value, absolute
   difference, and relative difference on rank zero.
3. Repeat with 1 and 32 MPI ranks. Sort by global DOF ID and require identical
   IDs and coordinates across paths/rank counts.
4. Require finite values, exactly `4 * global_owned_cells` unique IDs, no
   duplicates, and MPI-invariant direct values within `atol=1e-10,
   rtol=1e-8` (or a documented solver-precision bound).
5. Do **not** require the DG1 and direct values to match pointwise: DG1 is a
   smoothing projection. Failure means a compiler error, nonfinite value,
   missing/duplicate/moved material point, MPI-dependent direct value, or an
   unexplained discrepancy at an analytically constant test field.
