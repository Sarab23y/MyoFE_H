# Mechanical validation of the passive Fibrosis model

## Purpose

The tools in `validation/` test the current `Xi-ground-collagen` passive law at
two levels:

1. homogeneous material-point biaxial and shear loading;
2. pressure-controlled, passive inflation of the current LV mesh.

They do not modify production JSON files or parameters. Material tests call
the production `Forms._passive_energy_components` and
`Forms._passive_stress_components` methods, so the equations are not copied
into a second constitutive implementation. Inflation constructs the production
`LV_simulation` mesh and weak form, then uses its pressure-loading residual
without advancing the cardiac time loop.

## Files

| File | Role |
| --- | --- |
| `validation/material_point.py` | Production-backed homogeneous evaluator, pressure elimination, CSV/metadata writer. |
| `validation/biaxial_material_test.py` | Three prescribed stretch paths. |
| `validation/shear_material_test.py` | Local fiber-sheet and fiber-normal simple shear paths. |
| `validation/verify_material_validation.py` | Reference, decomposition, traction, and finite-difference checks. |
| `validation/passive_lv_inflation.py` | Adaptive quasi-static pressure sweep on the LV mesh. |
| `validation/configs/material_validation.json` | Example material-test ranges and output location. |
| `validation/configs/passive_inflation.json` | Example pressure targets, solver tolerances, volume conversion, and reference assumption. |
| `postprocessing/plot_material_validation.py` | Interactive or headless biaxial/shear plots. |
| `postprocessing/plot_passive_inflation.py` | Interactive or headless passive pressure-volume plot. |

All result paths are under `validation/results/` by default and are separate
from ordinary simulation output.

## 1. Material-point formulation

### Production law reuse

The evaluator creates a unit-volume DOLFIN mesh solely to evaluate homogeneous
UFL expressions. It subclasses `Forms` only to supply a prescribed
\(C=F^T F\). Constituent energies and stresses still come from the production
methods. The current law uses one shared elastic tensor for all three
constituents; no independent constituent deposition deformation gradients are
present.

The weighted material energy is

\[
W=\phi_g W_g+\phi_m W_m+\phi_c W_c.
\]

The output columns `energy_ground`, `energy_myofiber`, and
`energy_collagen` already include their corresponding fractions. Their sum is
`energy_total`.

### Incompressibility and stress convention

The prescribed material tests enforce \(J=1\). Production constituent PK2
stresses are first evaluated without incompressibility pressure. They are
pushed forward as

\[
\sigma_i=F S_i F^T/J.
\]

A **single** pressure multiplier is then chosen from the traction-free local
sheet-normal condition:

\[
p=(\sigma_g+\sigma_m+\sigma_c)_{nn},\qquad
\sigma=\sum_i\sigma_i-pI,
\]

so \(\sigma_{nn}=0\). The corresponding physical PK2 stress is

\[
S=\sum_i S_i-pJ C^{-1}.
\]

The pressure term is not assigned independently to each constituent. Thus the
constituent stress columns are energetic, fraction-weighted contributions;
the total columns add the one constraint pressure. This convention exactly
reconstructs total physical stress while avoiding an arbitrary allocation of
pressure between constituents.

### Reported measures

- `lambda_f`, `lambda_s`: directional stretches inferred from \(C\);
- `E_ff`, `E_ss`, `E_fs_tensor`: Green-Lagrange strain components;
- `engineering_shear_gamma`: prescribed engineering shear;
- `S_*`: second Piola-Kirchhoff stress;
- `sigma_*`: Cauchy stress;
- `J`: determinant of \(F\);
- `traction_normal_residual`: total \(\sigma_{nn}\), expected near zero;
- weighted constituent and total energies.

Stress and energy-density columns use the production internal unit. For the
current parameter input this is treated as Pa, but no conversion is performed
by the material scripts.

## 2. Biaxial tests

All biaxial paths prescribe

\[
F=\mathrm{diag}(\lambda_f,\lambda_s,
                 1/(\lambda_f\lambda_s)).
\]

The explicit paths are:

| Name | Definition | Meaning |
| --- | --- | --- |
| `equal_biaxial` | \(\lambda_f=\lambda_s=L\) | Equal prescribed stretches. |
| `fiber_dominant` | \(\lambda_f=L,\ \lambda_s=1+0.5(L-1)\) | Fiber stretch rises twice as fast as sheet stretch. |
| `sheet_dominant` | \(\lambda_s=L,\ \lambda_f=1+0.5(L-1)\) | Sheet stretch rises twice as fast as fiber stretch. |

These are **stretch-controlled paths**, not stress-controlled biaxial
protocols. The sheet/fiber stress ratio is an output, not a prescribed
boundary condition.

Run:

```bash
python validation/biaxial_material_test.py \
  validation/configs/material_validation.json
```

Expected files are `biaxial_<path>.csv` and
`biaxial_<path>_metadata.json` for each path.

### Interpretation

- Ground matrix responds through \(I_1\).
- Xi responds only when fiber stretch exceeds one.
- Collagen directions respond only when their corresponding \(I_4>1\).
- The normal collagen direction is inactive in these tensile in-plane paths
  because incompressible thickness contraction gives \(I_{4cn}<1\).
- There is no explicit fiber-sheet coupling term in the current law. A zero
  missing coupling contribution is therefore expected, not a validation
  failure.

An isolated Xi constituent has no sheet stiffness; directional collagen
constituents likewise do not form a general three-dimensional solid alone.
The tools report contributions at the same mixture deformation instead of
inventing stabilizing stiffness for isolated constituents.

## 3. Shear tests

The configured paths use local material axes \((f,s,n)=(x,y,z)\):

- `fiber_along_sheet`: \(F=I+\gamma e_f\otimes e_s\);
- `sheet_along_fiber`: \(F=I+\gamma e_s\otimes e_f\);
- `fiber_along_normal`: \(F=I+\gamma e_f\otimes e_n\).

All have \(J=1\). They activate the ground-matrix \(I_1\) response and can
stretch one directional collagen invariant even when an aligned biaxial path
does not. They do not reveal an explicit `I8fs`/fiber-sheet coupling response,
because no such production term exists.

Run:

```bash
python validation/shear_material_test.py \
  validation/configs/material_validation.json
```

Outputs are `shear_<direction>.csv` and matching metadata JSON files. Plot
`S_fs` against engineering \(\gamma\); `E_fs_tensor` is one half of
engineering shear only in the infinitesimal/simple interpretation, so the CSV
retains both definitions explicitly.

## 4. Numerical material verification

Run:

```bash
python validation/verify_material_validation.py \
  validation/configs/material_validation.json
```

The check verifies:

- zero constituent energy at \(F=I\);
- \(J=1\) and zero normal-traction residual;
- exact weighted energy reconstruction;
- a central finite-difference energy derivative against total PK2 stress on
  an isochoric fiber stretch path;
- a central finite-difference energy derivative against the appropriate
  first-Piola shear work conjugate;
- positive ground and collagen shear energies for the selected shear state.

This is a constitutive check, not experimental validation or parameter
calibration.

## 5. Passive LV inflation

### What the driver does

`passive_lv_inflation.py` loads an unmodified production instruction JSON and
constructs `LV_simulation`, thereby reusing:

- the same HDF5 mesh and facet boundaries;
- `f0/s0/n0` and quadrature fields;
- the same Xi-ground-collagen parameters and fractions;
- the same passive weak form, incompressibility, basal restraint, rigid-body
  constraints, and optional pericardial spring;
- the production cavity-volume integral.

It does **not** call `run_simulation` or `implement_time_step`. Circulation is
therefore not advanced, baroreflex controls are not advanced, growth is not
advanced, fiber reorientation is not advanced, and MyoSim ODEs are not
advanced. It sets the DOLFIN `cb_number_density` field to zero, making the
symbolic cross-bridge active stress identically zero during inflation, and
restores the local field before exit.

The pressure-controlled residual reuses production `F1` (passive), `F2`
(present but zero active stress), `F3_p` (prescribed endocardial pressure), and
`F4` (rigid constraints). It omits the volume-control residual `F3`. Because
the production mixed space still contains the now-unused cavity multiplier,
the validation residual constrains that one Real unknown to zero; this removes
an algebraic null row without altering displacement mechanics.

### Pressure and volume

The production circulation uses

\[
p_{\mathrm{mmHg}}=0.0075\,p_{\mathrm{internal}}.
\]

The driver therefore divides prescribed mmHg by `0.0075` before setting the
production pressure expression. DOLFIN's cavity-volume `assemble` is
collective, so the returned volume is global rather than a sum of already
global values.

The CSV records both model volume and `cavity_volume_ml`. The latter uses the
configuration's explicit `volume_scale_to_ml`. The example value is 1000,
consistent with interpreting current circulation volumes as litres, but it
must be verified for the selected mesh before publication.

### Continuation and failures

Requested pressures are traversed in increasing order. Each converged
displacement/multiplier vector becomes the initial guess for the next state.
If Newton raises `RuntimeError`, the last converged vector is restored and the
pressure interval is bisected. Failed attempts are retained in the CSV. If
half the failed interval is below `minimum_pressure_increment_mmHg`, the CSV
and metadata are written and the driver raises a clear terminal error; it does
not label the failed target as converged.

Run (rank count is an example, not a convergence recommendation):

```bash
mpiexec -np 32 python validation/passive_lv_inflation.py \
  validation/configs/passive_inflation.json
```

Expected files:

- `validation/results/passive_inflation/passive_inflation.csv`;
- `validation/results/passive_inflation/passive_inflation_metadata.json`;
- `validation/results/passive_inflation/input_instruction_snapshot.json`.

Plot:

```bash
python postprocessing/plot_passive_inflation.py \
  validation/results/passive_inflation/passive_inflation.csv --show

python postprocessing/plot_passive_inflation.py \
  validation/results/passive_inflation/passive_inflation.csv \
  --headless --save \
  validation/results/passive_inflation/passive_inflation.png
```

Interactive display occurs only with `--show`; figure creation occurs only
with `--save`. `--headless` selects Matplotlib's non-interactive `Agg` backend.

### Reference-state limitation

The example explicitly treats the supplied HDF5 geometry as a zero-pressure
reference. The driver performs no inverse unloading and introduces no
prestress. If the mesh represents an in-vivo loaded state, the resulting curve
is a model inflation curve from that geometry—not a validated unloaded-organ
EDPVR. This assumption is recorded in metadata.

The curve can be compared with an experimental EDPVR only after matching the
reference configuration, boundary/pericardial conditions, pressure and volume
units, and passive parameter context. A filling segment from one ordinary
closed-loop PV cycle is not an EDPVR; it normally provides one end-diastolic
point.

## 6. Plotting material results

```bash
python postprocessing/plot_material_validation.py \
  validation/results/material --show

python postprocessing/plot_material_validation.py \
  validation/results/material --headless \
  --save validation/results/material/material_validation.png
```

Biaxial plots show total PK2 stress versus Green-Lagrange strain for fiber and
sheet curves, plus total weighted energy. Shear plots show total `S_fs` versus
engineering shear and total energy. Constituent columns remain available in
CSV for custom plots and fair comparisons at the same deformation.

## 7. Study-quality verification still required

Before interpreting fitted or experimental agreement:

1. **Pressure-step sensitivity:** repeat inflation with successively smaller
   requested increments and compare volume at common pressures. Adaptive
   convergence alone does not establish path resolution.
2. **Mesh convergence:** repeat with at least two refined meshes while holding
   geometry, fibers, material parameters, loading, and boundary conditions
   consistent. Compare the whole curve, not one point.
3. **Solver tolerance sensitivity:** verify that tighter nonlinear tolerances
   do not materially change the curve.
4. **Reference configuration:** establish whether the mesh is unloaded or
   prestressed. Do not infer this from an HDF5 filename.
5. **Units:** independently confirm mesh length/cavity-volume units and the
   configured conversion to mL.
6. **Boundary conditions:** assess whether basal fixation and any pericardial
   spring match the experiment being compared.
7. **MPI equivalence:** compare a small serial run with two ranks using the same
   mesh and pressure points. The production mesh is available, but this review
   environment does not provide the legacy DOLFIN runtime needed to perform
   that comparison.

## 8. What has and has not been tested here

Static syntax, JSON parsing, configuration preservation, and source-level
reuse can be checked in a generic development environment. Executing the UFL
material evaluator and passive inflation requires the repository's legacy
FEniCS/Python 2 environment. No full cardiac run, large sweep, mesh refinement,
pressure-step convergence study, or experimental comparison is performed by
these tools automatically.
