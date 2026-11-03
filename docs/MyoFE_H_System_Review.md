# MyoFE_H current-system technical review

## 1. Scope and evidence

This document describes the current working-tree implementation as a Fibrosis
project. It does not treat archived demos, `python_codes_Sara`, old comments,
or former Fiber Disarray experiments as authoritative. Those files were
inspected only to identify repository organization and legacy risk.

The review covered the maintained entry point, current base JSON and its mesh,
the LV driver, weak-form and constitutive code, nonlinear solver, circulation,
heart rate, half-sarcomere kinetics, optional baroreflex/growth/reorientation,
MPI mappings, CSV/NPY/XDMF writers, plotting utilities, tests, and job scripts.
No production simulation or output regeneration was performed.

## 2. What the current system is

MyoFE_H couples four principal models:

1. **Three-dimensional LV mechanics.** A tetrahedral DOLFIN mesh is solved
   with a mixed finite-element formulation for displacement, hydrostatic
   pressure, cavity-volume pressure, and rigid-body constraints.
2. **Fibrosis passive mechanics.** Ground matrix, Xi myofiber, and
   three-direction tension-only collagen energies are independently
   parameterized and fraction weighted.
3. **Active cellular mechanics.** One half-sarcomere object is maintained for
   every locally owned scalar quadrature point. Calcium and myofilament ODEs
   produce cross-bridge stress, which becomes a fiber-aligned active stress in
   the FE weak form.
4. **Closed-loop circulation.** A four- or six-compartment lumped circulation
   advances blood volumes and supplies the target LV cavity volume. The FE
   cavity-pressure multiplier returns LV pressure to the circulation.

Optional baroreflex, infarct, growth/remodeling, and fiber-reorientation paths
are selected by JSON sections. Their presence in the source does not mean they
are active in the current baseline.

## 3. Architecture and responsibilities

| Module | Class/function | Implemented responsibility |
| --- | --- | --- |
| `python_codes/MyoFE.py` | `MyoFE`, `execute_MyoFE` | Create `MPI.COMM_WORLD`, parse/recode JSON, instantiate the LV model, run it. |
| `LV_simulation/LV_simulation.py` | `LV_simulation.__init__` | Construct template/local half-sarcomeres, mesh/weak form, solver, quadrature mappings, circulation, heart rate, optional modules. |
| same | `run_simulation` | Allocate outputs, configure XDMF, iterate over time, coordinate failure flushing, finalize outputs. |
| same | `implement_time_step` | Advance circulation, controls, cellular ODEs, FE solve, stress/length feedback, growth/reorientation, and saved state. |
| same | `create_data_structure*` | Select and allocate scalar and spatial output containers. |
| same | `write_complete_data_*`, `handle_output` | Extract local data, MPI-gather it, validate global order, and write CSV/NPY. |
| `mesh/mesh.py` | `MeshClass.__init__` | Load mesh and fields, initialize spaces/functions, validate Fibrosis input, build weak form. |
| same | `initialize_function_spaces` | Create mixed CG/Real mechanics spaces and scalar/vector/tensor Quadrature spaces, then configured DG/Quadrature spaces. |
| same | `initialize_functions` | Read facet markers and `f0/s0/n0`; create state, stress, coordinate, and heterogeneous-parameter functions. |
| same | `report_mesh_diagnostics` | Read-only rank-zero summary of owned cells, unique vertices, quadrature DOFs, and vector/coordinate consistency. |
| same | `create_weak_form` | Build passive/active/cavity/rigid-body residuals and consistent Jacobians. |
| `dependencies/fibrosis_config.py` | `validate_xi_ground_collagen` | Require nested constituent parameters and fractions, reject invalid sums, return resolved values without mutation. |
| `dependencies/forms.py` | `Forms` | Kinematics, constituent energies/stresses, incompressibility, cavity volume/pressure, strain helpers. |
| `dependencies/nsolver.py` | `NSolver.solvenonlinear` | Newton iterations for the current mechanics residual/Jacobian. |
| `half_sarcomere/half_sarcomere.py` | `half_sarcomere` | Couple membrane calcium, myofilament kinetics, movement, and stress at one material point. |
| `half_sarcomere/membranes/membranes.py` | `membranes` | Current simple two-compartment calcium model; other membrane files are not selected by the current base JSON. |
| `half_sarcomere/myofilaments/kinetics.py` | `evolve_kinetics`, `return_fluxes` | Three-/four-state SRX kinetic ODEs and rate fluxes. |
| `half_sarcomere/myofilaments/forces.py` | stress functions | Cross-bridge and intracellular/extracellular passive half-sarcomere stresses. |
| `circulation/circulation.py` | `Circulation` | Compartment initialization, valve-aware flows, SciPy volume integration, pressure update. |
| `heart_rate/heart_rate.py` | `heart_rate` | Beat timing, activation pulse, end-diastolic flag, heart-rate reporting. |
| `baroreflex/baroreflex.py` | `baroreflex`, `reflex_control` | Optional afferent balance and controlled parameter updates. |
| `growth/growth.py` | `growth`, `growth_component` | Optional setpoint/stimulus accumulation, growth multipliers, reference-mesh update. |
| `fiber_reorientation/fiber_reorientation.py` | `fiber_reorientation` | Optional stress-driven fiber update; absent from current base input. |
| `display/multi_panel.py` | `multi_panel_from_flat_data` | Template-driven plots from CSV/Excel data. |

## 4. Input-to-output execution trace

### 4.1 Input loading

`MyoFE.py` expects `MyoFE.py LV_sim <instruction.json>`. It reads the specified
file with `json.load`, then `recode` recursively converts Unicode strings to
Python 2 byte strings. It does not merge the current input with a master
default JSON. Model constructors then copy values from one-item lists using
`[0]`.

Constitutive input has additional explicit validation. For
`Xi-ground-collagen`, every constituent coefficient and `phi_m/phi_g/phi_c`
must exist, be a one-item scalar list, and yield a fraction sum close to one.
The validation result is printed on rank 0 before weak-form construction.

### 4.2 Mesh and function spaces

`MeshClass` joins the process working directory to `mesh.mesh_path[0]`, opens
the file through DOLFIN `HDF5File`, and reads `ellipsoidal`. The current code
does not branch on the JSON `relative_path` flag; process working directory is
therefore operationally important.

The internally created mechanics space is:

- displacement: vector CG2;
- incompressibility pressure: scalar CG1;
- LV cavity multiplier: Real0;
- five rigid-motion multipliers: Real0.

It also creates scalar Quadrature degree 2 for MyoSim, a mixed Quadrature
population space, and a 3×3 tensor Quadrature space. Configured spaces add the
DG scalar stimulus/output space, DG growth tensor space, vector Quadrature
material-coordinate space, and scalar Quadrature space. The historic input
spelling `scaler` is explicitly interpreted as scalar, with a `scalar` alias.

### 4.3 Material directions and material points

`initialize_functions` reads `f0`, `s0`, and `n0` from HDF5 into the configured
vector Quadrature degree-2 space. It also reads or constructs `ell`, `err`,
`ecc`, endocardial/epicardial distances, facet markers, half-sarcomere fields,
kinetic population fields, and heterogeneous scalar parameters.

The current normal initialization does not call
`apply_static_fiber_architecture`; therefore the old correlated disarray and
random-rotation methods remain source code but do not alter the baseline
`f0/s0/n0` path.

For the reported mesh:

| Quantity | Global/current reported value |
| --- | ---: |
| Cells | 1,252 |
| Unique vertices | 442 |
| Quadrature degree | 2 |
| Quadrature points per tetrahedral cell | 4 |
| Scalar material points | 5,008 |

Rank-0 direction vectors have a flat local length of 456, logically 152×3.
This matches 38 locally owned cells × 4 quadrature points. The count is local;
global physical vectors number 5,008 for each direction.

### 4.4 Weak form

`create_weak_form` builds `Forms(params)` from mesh/function data and the
validated passive-parameter functions. In the current baseline:

\[
F=I+\nabla u,\qquad F_e=F,\qquad C_e=F_e^T F_e,\qquad J=\det F_e.
\]

When optional growth supplies `Fg`, `Forms.Fe` instead uses
\(F_e=F F_g^{-1}\). There is one shared elastic tensor for the three passive
constituent energies; there are no separately stored `Fe_g`, `Fe_m`, and
`Fe_c` tensors in the current implementation.

The residual contains:

- `F1`: derivative of passive energy, including mixed incompressibility;
- `F2`: active fiber stress transformed with total `F` to the weak form;
- `F3`: LV cavity-volume constraint with its Real multiplier;
- `F4`: rigid-body constraints;
- optional pericardial spring contribution.

The Jacobian is obtained by differentiating these residual terms with respect
to the mixed unknown.

### 4.5 Initialization of cell-scale and circulation state

The LV driver creates one independent half-sarcomere object per local
quadrature point. Heterogeneous DOLFIN parameter functions can overwrite
selected local half-sarcomere parameters (`k_1`, `k_3`, `k_on`,
`cb_number_density`, `k_cb`, `x_ps`). Local and global quadrature DOF maps are
constructed for later output assembly.

The circulation initializes each vascular compartment at slack volume, uses
the FE reference cavity volume for the ventricular slack/current volume, and
places remaining blood volume in the veins. Vascular pressure is
`(volume-slack_volume)/compliance`; LV pressure comes from the FE cavity
multiplier and is multiplied by `0.0075` for reporting/use with the circulation.

### 4.6 One time step

The implemented order is:

1. Integrate compartment volumes with `solve_ivp`; valve logic prevents
   forward flow when the relevant pressure gradient is reversed, except for
   configured insufficiency conductance.
2. Evaluate the optional baroreflex activation window and update controlled
   parameters if a baroreflex object exists.
3. Apply configured protocol perturbations and infarct changes.
4. Rebuild any arrays affected by perturbations.
5. Advance heart-rate timing and obtain activation/new-beat/end-diastole flags.
6. At each local quadrature point, move cross-bridge distributions for length
   change, integrate membrane calcium, integrate myofilament populations, and
   update force/state data.
7. Copy population, parameter, and old-length arrays into DOLFIN functions.
8. Set the FE target cavity volume to the circulation's ventricular volume.
9. Solve the nonlinear mechanics equations.
10. Recover LV pressure, new half-sarcomere lengths, length increments,
    cross-bridge stress, and non-negative Xi passive-stress feedback.
11. Update scalar/circulation data and optional growth/reorientation behavior.
12. At the configured frequency, store spatial/scalar data and write selected
    mesh fields.

### 4.7 Nonlinear solve

`NSolver` starts with relative and absolute tolerances of `1e-7` and a maximum
of 50 iterations, overridden only by `mesh.solver.params` if present. Its
normal path uses DOLFIN Newton solve when `debugging_mode` is false. The source
also contains a manual/debug Newton implementation with extensive diagnostic
branches. The exact linear solver depends on the active branch and installed
DOLFIN/PETSc configuration.

### 4.8 Output finalization

Each rank stores local quadrature-point results. `handle_output` sends fields
from non-root ranks to rank 0. For non-averaged output, rank 0 places each
rank's values into global quadrature-DOF columns using `dofmap_list`. For
averaged output, it combines rank means weighted by local integration-point
count.

Rank 0 writes scalar `data.csv`; averaged runs write `spatial_data.csv`, while
non-averaged runs write one CSV per field. Non-averaged runs additionally run
global coverage/order diagnostics and may write canonical NPY arrays. XDMF
fields are written during time stepping through a collective DOLFIN file.

## 5. Constitutive and active material implementation

### 5.1 Passive constituent table

| Constituent | Current energy/stress behavior | Parameters | Fraction |
| --- | --- | --- | --- |
| Ground matrix | \(\Psi_g=\frac{a_g}{2b_g}[\exp(b_g(I_1-3))-1]\) | `ground_matrix.a_g`, `b_g` | `phi_g` |
| Myofiber | \(\Xi=c_3(\lambda_m-1)^2\) for \(\lambda_m>1\), else zero; \(\Psi_m=c_2(\exp(\Xi)-1)\). Stress retains the pre-existing closed form. | `myofiber.c2`, `c3` | `phi_m` |
| Collagen-f | Exponential in \((\max(I_{4cf},1)-1)^2\) | `a_cf`, `b_cf` | `phi_c` |
| Collagen-s | Exponential in \((\max(I_{4cs},1)-1)^2\) | `a_cs`, `b_cs` | `phi_c` |
| Collagen-n | Exponential in \((\max(I_{4cn},1)-1)^2\) | `a_cn`, `b_cn` | `phi_c` |
| Incompressibility | Current weak form uses `-p*(J-1)`; code retains a `Kappa/2*(J-1)^2` alternative when `incompressible` is false. | pressure unknown or `Kappa` | Not a material fraction |

Ground and collagen PK2 stresses are obtained by UFL differentiation with
respect to a variable `Cmat`. Xi stress is calculated in its retained closed
form and rotated from the local fiber basis. Total passive PK2 stress is the
sum of fraction-weighted ground, myofiber, collagen, and incompressibility
terms.

### 5.2 Active mechanics

Myofilament populations generate cross-bridge stress from bound-bin
populations, cross-bridge density/stiffness, power-stroke offset, and
half-sarcomere length change. `k_on` enters thin-filament activation through
calcium, overlap, unactivated sites, and cooperativity. The FE active tensor is
currently purely fiber aligned: `cb_stress * outer(f0,f0)`; sheet and normal
fractions are hard-coded to zero.

### 5.3 Distinct passive-stress concepts

The code contains two related passive mechanisms that should not be conflated:

- the FE Xi-ground-collagen passive law supplies tissue mechanics;
- `half_sarcomere/myofilaments/forces.py` also defines intracellular and
  extracellular one-dimensional passive functions. In the coupled step, the
  projected FE Xi `Sff` is fed back as `pass_stress_list` after negative values
  are clipped. The standalone half-sarcomere passive functions remain in the
  module but are not the source of the current FE passive residual.

## 6. Coordinate systems, stress quantities, and units

| Quantity | Current representation | Meaning established from code |
| --- | --- | --- |
| `f0`, `s0`, `n0` | Vector Quadrature degree 2 | Reference fiber, sheet, and sheet-normal directions loaded from HDF5. |
| `ell`, `err`, `ecc` | Vector Quadrature degree 2 | Longitudinal, radial, and circumferential local-coordinate directions used for strain/output. |
| `Pactive` | UFL 3×3 tensor | Fiber-aligned active PK2-type contribution used as `inner(F*Pactive, grad(v))`. |
| `passive_total_stress` | UFL 3×3 tensor | Sum of fraction-weighted material PK2 terms and incompressibility. |
| `myo_passive_PK2` | UFL 3×3 tensor | Fraction-weighted Xi myofiber passive component. |
| `bulk_passive` | UFL 3×3 tensor | Current returned matrix component (ground plus collagen), despite the legacy label `bulk`. |
| `incomp_stress` | UFL 3×3 tensor | Pressure/penalty contribution kept separate from ground-matrix material. |
| `Ell`, `Err`, `Ecc` | Scalar UFL expressions | Green-strain projections onto local longitudinal/radial/circumferential directions. |

Verified unit evidence is limited:

- `0.0075 * LVcavitypressure()` is explicitly described and used as a
  conversion to mmHg. This is consistent with an underlying pressure/stress
  scale close to pascals.
- The Fibrosis parameter note identifies stress-like coefficients as pascals
  after conversion from literature kPa.
- Time is treated as seconds in protocol names and integration intervals.
- Half-sarcomere lengths and cross-bridge offsets are consistent with
  nanometre-scale formulas (`1e-9` appears in cross-bridge stress), but there is
  no central unit declaration.
- Geometry, cavity volume, blood-volume, resistance, compliance, flow, and CSV
  unit conventions cannot all be proven from the repository. They need a
  documented dimensional audit against mesh-generation settings and a known
  validated run.

## 7. Output review

### 7.1 Selection behavior

`save_outputs: all` does not enumerate every field. It sets the requested list
to empty, and an empty list triggers a maintained default set. The final
spatial container deliberately includes only `spatial_fiber_data_fields`,
`spatial_extra`, and optional growth fields; lists of all half-sarcomere,
myofilament, and membrane fields are populated but excluded from `data_field`.

The default includes:

- `f01/f02/f03`, but not `s01..s03` or `n01..n03`;
- `lx/ly/lz`, `ecc*`, `err*`, `ell*`, `endo_dist`;
- quadrature-sampled `dx/dy/dz`;
- active, selected passive, incompressibility, and strain quantities;
- `hs_length`, `cb_number_density`, and `k_1`.

Thus **“all” means the maintained default output contract, not all supported
or internal state**.

### 7.2 Spatial mapping

The current writer projects displacement from vector CG2 into the same vector
Quadrature degree-2 space used by material directions, then separates XYZ
components. Scalar active stress and half-sarcomere length are projected into
the scalar Quadrature space. Coordinates come from that scalar space's DOF
coordinates.

For non-averaged MPI output, global columns follow scalar quadrature global
DOF IDs. Diagnostics reject duplicates, missing/out-of-range DOFs, inconsistent
field ordering, and coordinate-value mismatches. This is a stronger contract
than concatenating rank-local arrays.

### 7.3 Format relationships

- CSV and canonical NPY files use material/quadrature-point data.
- XDMF is used for mesh-associated visualization. Displacement remains nodal
  CG2; selected scalar fields are projected to the configured DG scalar space.
- These formats describe related physical states but do not all use the same
  FE support. It is incorrect to expect nodal displacement XDMF arrays and
  quadrature CSV arrays to have identical point counts.
- Canonical NPY files are only attempted for non-averaged output with required
  coordinate/basis fields present.

## 8. MPI behavior

Two communicator forms coexist: the application receives mpi4py
`MPI.COMM_WORLD`, while DOLFIN mesh operations expose a PETSc communicator.
The mesh diagnostic explicitly selects the mpi4py communicator for Python
collectives.

Owned-cell counting uses the topology ghost offset when available; global
vertices use DOLFIN's native `size_global(0)`; quadrature counts use ownership
ranges and global space dimension. In the reported run there were no ghost
cells, but the logic distinguishes owned and ghost entities.

The LV driver also builds `dofmap_list`, `int_points_per_core`, and global
quadrature counts for root assembly. Non-root ranks send spatial dictionaries
field by field. This may be memory- and communication-intensive for large
unaveraged studies because rank 0 constructs global pandas tables.

## 9. Review findings

### Finding 1 — no reproducible environment specification

- **File/function:** repository root; imports and SLURM scripts throughout.
- **Evidence:** no `requirements.txt`, Conda environment, container recipe, or
  packaged metadata exists; job scripts name site-specific images.
- **Consequence:** users cannot reproduce the exact legacy FEniCS/Python 2
  stack from the repository alone.
- **Status:** confirmed.
- **Next step:** record the working image digest and package versions, then add
  a container or environment manifest without upgrading scientific code in
  the same change.

### Finding 2 — `save_outputs: all` is not literal

- **File/function:** `LV_simulation.py`, `run_simulation` and
  `create_data_structure_for_spatial_variables`.
- **Evidence:** `all` selects an empty request list; the default creates many
  candidate lists but ultimately stores fiber/extra/growth fields only.
- **Consequence:** a user may expect membrane, myofilament, sheet, and normal
  components that are not written.
- **Status:** confirmed.
- **Next step:** define and document an explicit output registry; either rename
  the mode to `default` or make `all` enumerate every supported field with a
  storage warning.

### Finding 3 — current passive-material note is stale

- **File/function:** `docs/reviews/fibrosis_passive_material_parameters.md`.
- **Evidence:** its final sentence says the example keeps `phi_c = 0`; the
  current baseline has `phi_c = 0.03`, and the three fractions sum to one.
- **Consequence:** readers may misunderstand whether collagen contributes.
- **Status:** confirmed documentation defect; intentionally not modified in
  this review because the task requested preserving existing files while
  adding canonical documentation.
- **Next step:** correct or replace that note in a dedicated documentation
  cleanup after confirming desired provenance language.

### Finding 4 — base run constructs a baroreflex object but never reaches its window

- **File/function:** current base JSON; `LV_simulation.__init__`,
  `implement_time_step`; `protocol.baro_activation`.
- **Evidence:** `model.baroreflex` exists, so an object is constructed. The
  protocol activates it from 20 to 40 s, while 1,000 × 0.001 s covers about
  1 s.
- **Consequence:** the base run behaves without active baroreflex controls,
  but not because the baroreflex model is absent. Extending duration beyond
  20 s changes behavior.
- **Status:** confirmed from configuration/control flow; scientific intent
  requires user confirmation.
- **Next step:** label inputs explicitly as “model present/window inactive” or
  remove the section only in a separately authorized configuration change.

### Finding 5 — path semantics depend on launch directory

- **File/function:** `mesh.py`, `MeshClass.__init__`.
- **Evidence:** mesh path is joined to `os.getcwd()`; `relative_path` is not
  consulted there.
- **Consequence:** the same JSON may work from `python_codes` and fail from the
  repository root or a scheduler's default directory.
- **Status:** confirmed.
- **Next step:** resolve paths relative to the instruction file or formally
  document a fixed launch directory; this requires passing the input-file path
  into the model.

### Finding 6 — Python 2 CLI edge case

- **File/function:** `MyoFE.py`, `MyoFE`.
- **Evidence:** the `no_of_arguments == 2` branch reads `sys.argv[2]`, which is
  out of range. Normal documented invocation has three arguments and avoids
  it.
- **Consequence:** incomplete invocation can raise `IndexError` rather than a
  useful usage message.
- **Status:** confirmed.
- **Next step:** replace the manual branching with explicit argument validation
  or `argparse` in a separate code task.

### Finding 7 — material kinematics are shared, not constituent-specific

- **File/function:** `forms.py`, `Fe`, `Cmat`,
  `_passive_energy_components`.
- **Evidence:** ground, Xi, and collagen receive one `Cmat`; optional growth
  changes that common elastic tensor through one `Fg`.
- **Consequence:** documentation or calibration should not claim independent
  constituent deposition stretches/reference tensors.
- **Status:** confirmed implementation behavior; whether this is scientifically
  sufficient requires model-design review.
- **Next step:** document the intended constrained-mixture assumptions before
  considering any constituent-specific kinematic extension.

### Finding 8 — `bulk_passive` label combines ground and collagen

- **File/function:** `forms.py`, `stress`; spatial output mapping in
  `LV_simulation.py`.
- **Evidence:** `matrix_passive = ground_passive + collagen_passive` is returned
  through the legacy output slot consumed as `bulk_passive`.
- **Consequence:** post-processing may incorrectly interpret that field as
  ground matrix alone.
- **Status:** confirmed.
- **Next step:** add separately named ground/collagen outputs and retain a
  documented compatibility aggregate before deprecating `bulk_passive`.

### Finding 9 — active and passive `total_stress` is a PK2-like sum

- **File/function:** `mesh.py`, `create_weak_form`.
- **Evidence:** `total_stress = Pactive + passive_total_stress`, while the weak
  active term uses `F*Pactive`. Several output labels simply say “stress.”
- **Consequence:** interpreting outputs as Cauchy stress would be incorrect
  without an explicit push-forward.
- **Status:** confirmed for construction; some downstream naming remains
  ambiguous.
- **Next step:** establish a stress-measure naming convention (`PK2`, `PK1`,
  Cauchy) and record transformations in output metadata.

### Finding 10 — unit system is only partially documented

- **File/function:** circulation pressure conversion, material note,
  cross-bridge force code, JSON.
- **Evidence:** seconds and a `0.0075` pressure conversion are explicit, but no
  central definitions cover geometry, volume, flow, resistance, compliance,
  or every stress output.
- **Consequence:** combining meshes or parameters from other studies risks a
  silent scale mismatch.
- **Status:** confirmed documentation gap; actual units require external case
  provenance.
- **Next step:** create a versioned unit table verified against the mesh
  generator and one trusted baseline output.

### Finding 11 — archived trees can be mistaken for maintained code

- **File/function:** `python_codes_Sara`, `demos_n`, numerous copied/old forms
  and solver files.
- **Evidence:** parallel implementations and extensive historical JSON/job
  files coexist with the current path.
- **Consequence:** searches and edits can target inactive copies; users may run
  stale scientific behavior accidentally.
- **Status:** confirmed repository-maintenance risk.
- **Next step:** add an archive manifest and eventually move immutable history
  to a release/archive branch, after verifying no active workflows depend on
  it.

### Finding 12 — optional growth/reorientation paths retain legacy assumptions

- **File/function:** `growth/growth.py`, `fiber_reorientation.py`, growth and
  reorientation branches in `LV_simulation.py`.
- **Evidence:** these paths use separate mechanics classes, legacy function
  space names, hard-coded thresholds/constants, and extensive debug logic;
  they are absent from the current baseline and current tests do not execute
  them end to end.
- **Consequence:** enabling them with the current Fibrosis schema may expose
  interface or interpretation problems not seen in the baseline.
- **Status:** requires further runtime testing; the lack of baseline activation
  is confirmed.
- **Next step:** test each optional module with a minimal validated input and
  document its required spaces/fields before relying on production results.

### Finding 13 — rank-zero assembly may limit large outputs

- **File/function:** `LV_simulation.py`, `handle_output`.
- **Evidence:** every non-root rank sends each pandas-backed field to rank 0,
  which constructs global data holders and writes all files.
- **Consequence:** memory usage and serialized communication grow with number
  of material points, states, and fields.
- **Status:** confirmed architecture; practical limit requires benchmarking.
- **Next step:** benchmark representative studies, then consider parallel HDF5
  or distributed XDMF for large unaveraged output.

### Finding 14 — tests are not full FEniCS execution tests

- **File/function:** `tests/`.
- **Evidence:** current tests validate configuration, formulas, JSON values,
  and source patterns without assembling/running the complete distributed
  model.
- **Consequence:** API, MPI, HDF5, and output failures may appear only on the
  cluster.
- **Status:** confirmed.
- **Next step:** add a tiny one-rank and two-rank smoke mesh with one or two
  steps in the pinned environment.

## 10. Prioritized improvements

1. **Capture the executable environment** (exact container and versions).
2. **Add serial and two-rank smoke tests** covering mesh load, one Newton step,
   XDMF, CSV, and non-averaged global DOF assembly.
3. **Publish a verified unit/stress-measure contract** for every user-facing
   input and output.
4. **Make output selection explicit**, especially the meaning of `all` and the
   distinction between quadrature and nodal data.
5. **Correct stale documentation and ambiguous labels**, notably `phi_c = 0`
   and `bulk_passive`.
6. **Make path resolution deterministic** relative to the input file.
7. **Validate optional growth and fiber-reorientation paths independently**
   before using them with the current Fibrosis law.
8. **Separate maintained and archived material** without deleting useful
   mesh/fiber/output infrastructure.
9. **Plan a Python 3/FEniCS migration only after regression baselines exist**;
   combining migration with scientific changes would make validation harder.

## 11. What remains uncertain

The repository alone does not establish:

- the exact FEniCS, PETSc, MPI, Python, and system-library versions in the
  production image;
- the physical length and volume units of every mesh/circulation quantity;
- calibration provenance for all current material and kinetic values;
- whether every archived JSON is executable with the maintained code;
- ParaView compatibility across versions beyond DOLFIN's XDMF intent;
- numerical robustness of growth, infarct, reorientation, or alternative
  membrane/kinetic branches under the current Fibrosis model;
- scalability limits of root-gathered pandas output.

These uncertainties require environment metadata, trusted reference results,
or targeted runtime tests; they should not be resolved by inference from old
study names.
