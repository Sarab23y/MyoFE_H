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

A **single** pressure multiplier is then chosen to make the normal component
of traction zero on the deformed image of the local sheet-normal face. For
reference normal \(N=e_n\), the evaluator calculates the actual unit normal
\(n=F^{-T}N/\|F^{-T}N\|\) and sets:

\[
p=(\sigma_g+\sigma_m+\sigma_c)_{nn},\qquad
\sigma=\sum_i\sigma_i-pI,
\]

so \(n\cdot\sigma n=0\). The corresponding physical PK2 stress is

\[
S=\sum_i S_i-pJ C^{-1}.
\]

The pressure term is not assigned independently to each constituent. Thus the
constituent stress columns are energetic, fraction-weighted contributions;
the total columns add the one constraint pressure. This convention exactly
reconstructs total physical stress while avoiding an arbitrary allocation of
pressure between constituents. The CSV separately reports the three traction
components, normal residual, tangential magnitude, and full magnitude.

For aligned diagonal biaxial deformation, symmetry makes tangential traction
zero, so the thickness face is fully traction free. For prescribed simple
shear, the pressure eliminates only normal traction. Tangential traction can be
nonzero—and is required to maintain shear—so the documentation does not call
those surfaces fully traction free.

### Reported measures

- `lambda_f`, `lambda_s`: directional stretches inferred from \(C\);
- `E_ff`, `E_ss`, `E_fs_tensor`: Green-Lagrange strain components;
- `engineering_shear_gamma`: prescribed engineering shear;
- `S_*`: second Piola-Kirchhoff stress;
- `sigma_*`: Cauchy stress;
- `J`: determinant of \(F\);
- `traction_normal_residual`: \(n\cdot\sigma n\), expected near zero;
- `traction_tangential_magnitude` and `traction_full_magnitude`: distinguish
  zero normal stress from zero full surface traction;
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

The check evaluates the production UFL energy/stress equations and verifies:

- zero constituent energy at \(F=I\);
- \(J=1\) and zero normal-traction residual;
- exact weighted energy reconstruction;
- central finite-difference energy derivatives against \(P:dF/dq\), with
  \(P=FS\), on all three configured biaxial path definitions;
- the same work-conjugate derivative check on all three shear directions;
- positive ground and collagen shear energies for the selected shear state.

This is a constitutive check, not experimental validation or parameter
calibration. The generic unit tests only inspect source/configuration
structure; this script is the check that actually assembles and evaluates the
production numerical UFL equations and therefore must be run in FEniCS.

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
restores the local field before exit. Construction reads the mesh HDF5 file in
read-only mode and does not initialize the normal simulation output handler;
validation output is directed only to the configured validation directory.
With `overwrite: false`, an existing inflation CSV causes a clear error.

The pressure-controlled residual reuses production `F1` (passive and mixed
incompressibility), `F2`
(present but zero active stress), `F3_p` (prescribed endocardial pressure), and
`F4` (rigid constraints). It omits the volume-control residual `F3`. Because
the production mixed space still contains the now-unused cavity multiplier,
the validation residual constrains that one Real unknown to zero; this removes
an algebraic null row without altering displacement mechanics. The code also
checks across MPI ranks that the active `cb_number_density` coefficient is
exactly zero before solving.

The prescribed pressure has the same sign as production `F3_p =
Press*inner(J F^{-T}N,v)*ds(endo)`. On the endocardial boundary the solid's
outward normal points into the cavity; writing internal virtual work minus the
physical pressure traction produces this plus sign. This is also consistent
with the negative orientation in the production cavity-volume surface
integral. A smoke run must still verify that positive pressure increases
cavity volume for the selected facet markers.

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
configuration's required, explicit `volume_scale_to_ml`. No default is
provided: the example value is `null`, and the driver refuses to run until a
positive value is supplied.

This is necessary because cavity volume is the integral of the mesh-coordinate
geometry and therefore has units of coordinate-unit cubed. The checked-in base
HDF5 file is only a Git LFS pointer in this environment, so its coordinates
cannot be inspected. Available mesh-generation files also contain different
dimensionless-looking scale values (including 1.2 and 11) without a definitive
unit declaration tying either generator to this exact HDF5 object. A plotting
template labels historical cavity volume as litres, but neither that label nor
the circulation's blood-volume convention proves the mesh coordinate unit.

### Continuation and failures

Requested pressures are traversed in increasing order. Each converged
displacement/multiplier vector becomes the initial guess for the next state.
If Newton raises `RuntimeError`, all ranks first agree on failure through an
MPI all-reduction. The complete mixed vector `w`—displacement, hydrostatic
pressure, cavity multiplier, and rigid-body multipliers—is restored from the
last converged distributed vector and synchronized with `apply('insert')`.
There are no time-evolved material history variables in this driver; pressure
is reset explicitly for every attempt and active density remains zero. The
pressure interval is then bisected. Failed attempts are retained in the CSV. If
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

The example *declares an assumption* that the supplied HDF5 geometry is the
zero-pressure reference. It is not verified. The production main mechanics
uses `F=I+grad(u)` and, unless a growth tensor is explicitly passed to `Forms`,
uses `Fe=F`; ordinary `MeshClass` construction does not load prestress or a
constituent-specific deposition state. Initial `hsl0` and `f0/s0/n0` fields are
loaded, but at `F=I` the Xi stretch ratio is one.

If the input contains a growth module, its separate mechanics object can be
constructed, but the inflation driver never advances or transfers a growth
update into the primary reference configuration. Consequently, a previously
grown state affects inflation only if it has already been exported as the
selected input mesh/fields. The driver performs no inverse unloading and
introduces no prestress. If the mesh represents an in-vivo loaded state, the
resulting curve is a model inflation curve from that geometry—not a verified
unloaded-organ EDPVR. This assumption is recorded in metadata.

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

## 8. Cluster smoke-test script

`validation/run_validation_smoke.slurm` follows the repository's existing
Singularity/SLURM execution pattern without hard-coding a user account,
partition, repository path, image path, or unverified volume conversion. Set:

```bash
export FENICS_IMAGE=/path/to/the/working/fenics.img
export REPO=/path/to/MyoFE_H
export VOLUME_SCALE_TO_ML=<verified-positive-conversion>
cd "$REPO"
sbatch validation/run_validation_smoke.slurm
```

The job runs production-UFL material verification, 0–1 mmHg serial inflation,
the same two-rank inflation, a matching-pressure volume comparison, and a
headless inflation plot. It writes only under `validation/results/`. This is a
smoke test, not pressure-step or mesh-convergence evidence.

## 9. What has and has not been tested here

Static syntax, JSON parsing, configuration preservation, and source-level
reuse can be checked in a generic development environment. Executing the UFL
material evaluator and passive inflation requires the repository's legacy
FEniCS/Python 2 environment. No full cardiac run, large sweep, mesh refinement,
pressure-step convergence study, or experimental comparison is performed by
these tools automatically.

During the 2026-09-15 audit, all 29 lightweight repository tests passed, as
did Python compilation, JSON parsing, shell syntax, and diff-whitespace
checks. The audit container did not provide `dolfin` or `matplotlib`, and the
base HDF5 path resolved to a Git LFS pointer rather than mesh contents.
Consequently, no claim is made here that the UFL derivative checks, inflation
solve, serial/MPI volume comparison, or plot generation have passed
numerically. Run the cluster smoke-test above in the project's working legacy
FEniCS image before using validation results scientifically.
