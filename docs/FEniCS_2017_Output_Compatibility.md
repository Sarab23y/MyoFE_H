# FEniCS 2017 output compatibility investigation

## Confirmed global-output failure

`self.data` contains both scalar state and material-point arrays.  The prior
global-table constructor registered every key, so `fiber_stretch_raw` and
`fiber_stretch_tension_only` created scalar `data.csv` cells and the first
assignment raised `ValueError: setting an array element with a sequence`.
Global fields are now classified from their runtime value shape.  Only scalar
or zero-dimensional numeric values are accepted; a selected field changing
shape later is a fatal, named error.  Spatial fields remain available through
their existing `<field>_data.csv` paths in legacy and buffered modes.

This classification controls recording only.  It does not control expression
construction, FE mechanics, MyoSim, ECM calibration, growth, or fibrosis.

## FFC JIT failure assessment

The unavailable FFC compiler diagnostic prevents attribution to one generated
C++ expression.  Source tracing nevertheless found a new output-only direct
Quadrature projection of `sqrt(f0.C.f0)` executed every mechanics timestep.
The mechanics already evaluates `hsl = alpha_f*hsl0` at the same degree-two
material points.  The raw stretch recorder now obtains the mathematically
identical `hsl/hsl0`, with an explicit zero-reference guard.  This removes the
duplicate output-only JIT form, not the stretch calculation.

The optional mechanical-health sampler now reuses the already evaluated raw
stretch, HSL, HSL0, and Xi stress.  Remaining diagnostic expressions use the
same unforced `project(expression, quadrature_space)` pattern present in the
historical DOLFIN-2017 code, rather than forcing a compiler representation for
every expression.  Failures name the exact diagnostic field.  The separately
validated direct-quadrature ECM implementation and the production DG1 MyoSim
passive-stress transfer are unchanged.

These are compatibility improvements, not proof of the exact cluster JIT root
cause.  Confirmation requires the included 200-step LCC smoke job.

## Smoke-test coverage and limits

`run_ecm_output_smoke.slurm` runs matched 200-step legacy and buffered cases
with 32 ranks, then compares required global fields, spatial fields, row counts,
timestamps, headers, and both reference coordinates and global quadrature IDs.
It refuses missing fields and unverified mapping.  Legacy spatial rounding is
explicitly tolerated; global scalar comparisons remain tight.

At 0.2 seconds this smoke test exercises the first mechanics/output writes,
mechanical-health sampling, direct instantaneous ECM evaluation, MPI assembly,
partial buffered flush, and normal completion metadata.  It does **not** reach
baroreflex activation, three-cycle ECM reference calibration, growth events,
or collagen evolution.  Those require separate short targeted tests and the
prepared 42-second healthy validation after the smoke test passes.

## Pass/fail checklist

1. Both solver invocations exit zero and retain 201 records internally, with
   output rows at the configured frequency including the initial/final state.
2. No scalar/spatial shape error and no FFC JIT error occurs.
3. The comparison report has `passed: true`; no required field is missing.
4. `quadrature_dof.npy` and `quadrature_global_dof.npy` match exactly.
5. Buffered `output_metadata.json` has `complete: true` and correct committed
   global/spatial row counts.
6. Any failed solver, missing field, mapping mismatch, duplicate output path,
   or numerical difference beyond documented rounding returns nonzero.
