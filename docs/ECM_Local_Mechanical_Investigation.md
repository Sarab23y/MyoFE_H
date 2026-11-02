# ECM local-mechanics investigation

## What the existing data establish

The production kinematics define
`lambda_f = sqrt(f0 . C f0)`, `hsl = lambda_f*hsl0`, and
`E_ff = (lambda_f^2 - 1)/2`.  The historical output called `alpha_f` is the
tension-only value `max(lambda_f, 1)` and therefore cannot display
shortening.  It has deliberately not been renamed or redefined.  New output
uses `fiber_stretch_raw` for `lambda_f` and
`fiber_stretch_tension_only` for the historical clipped expression.

At the reported point 1005, 21.6/988.6 gives `lambda_f` approximately
0.02185 and hence `E_ff` approximately -0.49976.  Those three reported values
are mutually consistent with almost complete local shortening.  Xi is
tension-only, so its zero stress is also consistent.  This is evidence of an
abnormal local kinematic state, but it does **not** establish inversion:
neither the full deformation gradient nor `J=det(F)` was saved in the old
CSV data.  The global DOF number is partition-specific evidence unless it is
paired with the reference coordinate and cell/quadrature metadata.

The onset at 26.8--27.2 seconds predates the resistance ramp, so the ramp
cannot be its initiating cause.  A future run must record raw stretch, J,
local mixed pressure, the stress components, reference coordinate and cell
identity at the same quadrature point.  The optional mechanical-health
diagnostic does this sparsely without changing mechanics.

## The 39.35-second stress event

The incompressibility contribution is second Piola--Kirchhoff stress
`S_inc = -p C^-1`.  A large multiplier `p`, an ill-conditioned `C`, or both
can therefore generate very large values even when the ECM-only tensor does
not contain this term.  The reported neighboring signs are compatible with
an ill-conditioned reconstructed field, but the historical dataset cannot
decide among a genuine local state, a projection artifact and an indexing
error.  In particular, historical `bulk_passive` and `total_passive` files
may associate DG1 values with quadrature headers; equal column labels are not
proof of material-point correspondence.

The minimum evidence needed is the directly evaluated quadrature tuple
`(F, J, p, C^-1, lambda_f, hsl, S_inc, S_ground, S_collagen, S_Xi, S_active)`
at verified reference coordinates for times adjacent to 39.35 seconds.
Until then, the old extrema are suspicious output evidence, not proof of a
constitutive or nonlinear-solver failure.

## Minimal runtime validation

Submit `validation/run_ecm_quadrature_mpi.slurm`.  It runs the same
deterministic manufactured ground-plus-collagen PK2 fiber expression on one
and 32 ranks.  Each run must contain finite values and exactly one record per
physical quadrature coordinate.  The DG1/direct difference is reported but
is not required to vanish because DG1 projection is a smoothing operation.
The cross-rank test fails if a coordinate is missing/duplicated or if direct
values at matching coordinates differ by more than `1e-10`.

Only after `comparison.json` reports `passed: true` should the healthy case
be submitted, with that report supplied through
`ECM_QUAD_VALIDATION_REPORT`.  A failure indicates unsupported Quadrature
compilation, inconsistent distributed mapping, missing/duplicate points, or
non-deterministic direct values and blocks the 42-second case.

## Consequences for ECM calibration

Near-collapse at even a small number of points can redistribute local
ground/collagen stress while global pressure and volume remain stable, and
can inflate a pointwise or integrated field-change measure.  It is therefore
a plausible contributor to the unstable homeostatic field, but causality is
not established by the old files.  Collagen remodeling should remain off
until the direct-quadrature MPI test and the 42-second healthy validation
case resolve material-point mapping and local kinematic health.
