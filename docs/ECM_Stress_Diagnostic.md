# Passive ECM fiber-stress diagnostic

This diagnostic does **not** evolve collagen. It measures the material (second
Piola--Kirchhoff) stress obtained from the already calibrated extracellular
energy

\[
\Psi_{ECM}=\phi_g\Psi_g+\phi_c\Psi_c,\qquad
S_{ECM}=2\partial\Psi_{ECM}/\partial C.
\]

The production `Forms` implementation already differentiates the two weighted
energies independently. The diagnostic adds those tensors and projects
`f0 . S_ECM . f0`; therefore the ground and collagen fractions are each used
once. Xi stress, MyoSim active stress, and the incompressibility multiplier are
not assigned to ECM. Values have the same stress unit and PK2 measure as the
existing total-stress output.

## Healthy reference

Every time step contributes `stress * dt` to the current cardiac-cycle
integral. At 20 s, the local reference field is frozen as the pointwise mean of
the last three *complete* cycles. Local vectors are retained in their existing
quadrature ordering; no ALE operation or mechanics reconstruction occurs in
this case. Global min/mean/max values and stability norms use MPI collectives.

Before accepting the label "homeostatic", the code checks the three-cycle
relative span of cycle-mean LV pressure, LV volume, and heart rate, plus the
relative L2 change between successive ECM-stress fields. The initial validation
case uses an explicit 5% tolerance. This is an operational baseline-stability
criterion, not a calibrated physiological parameter. Failure aborts at 20 s;
the code neither extends the baseline nor permits the overload to contaminate
the reference.

After freezing, it reports

\[
\Delta S_{ff}^{ECM}=\bar S_{ff}^{ECM}-S_{ff,h}^{ECM},\qquad
D_c=\max(\Delta S_{ff}^{ECM},0).
\]

No normalization, gamma exponent, threshold, saturation, or collagen rate law
is applied. The fractions remain `(0.700, 0.274, 0.026)` and are checked after
each completed cycle.

## Run

Submit one continuous baseline-to-overload job:

```bash
sbatch validation/run_br_ecm_stress_diagnostic.sh
```

The job refuses to overwrite
`/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/ECM_stress_diagnostic`, validates
the schedule, runs one MPI simulation, and generates temporal and PV plots.
Spatial fields are written to `mesh_output/solution.xdmf` for ParaView. A full
legacy-FEniCS run is required to decide whether the baseline passes the 5%
criterion and whether the post-ramp ECM excess is mechanically plausible.

## Strict and provisional reference modes

`baseline_stability_mode` defaults to `strict`. In strict mode, any pressure,
volume, heart-rate, or integrated ECM-field change above the configured 5%
tolerance writes `ecm_baseline_stability.json` and stops calibration. The
independent `ECM_40s_provisional` case selects `warn`: the same calculations,
PASS/FAIL output, failed-criterion names, and pointwise change hotspots are
retained, but a finite three-cycle reference is frozen for diagnostic use.
Missing cycles, nonfinite values, inconsistent local field sizes, and MPI/FE
integration failures remain fatal in both modes.

The CSV status columns distinguish existence from validation:
`ecm_reference_frozen`, `ecm_baseline_stable`, and
`ecm_reference_provisional`. Before calibration,
`ecm_excess_valid == 0`; signed and positive excess fields contain finite zero
placeholders solely for PETSc/XDMF safety. The plotting tool masks those
placeholders and only displays excess stress after the reference is frozen.
