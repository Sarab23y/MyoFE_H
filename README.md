# MyoFE_H: coupled left-ventricular fibrosis mechanics

MyoFE_H is a research code for simulating a deforming left ventricle (LV)
coupled to lumped circulation and spatially distributed half-sarcomere
kinetics. The maintained baseline is a **Fibrosis** model. Its passive
mechanics separate ground matrix, myofiber, and collagen constituents while
retaining generic mesh, cardiac-fiber, quadrature-point, and output
infrastructure developed during earlier work.

This repository is research software rather than a packaged Python library.
It contains current simulation code alongside many archived cases and code
copies. For the maintained execution path, start with `python_codes/MyoFE.py`
and `demos/base/sim_inputs/base_instruction.json`.

For a detailed implementation audit, see
[`docs/MyoFE_H_System_Review.md`](docs/MyoFE_H_System_Review.md).

## Repository map

| Path | Current role |
| --- | --- |
| `python_codes/MyoFE.py` | Command-line entry point; loads JSON, creates the MPI communicator, and launches the LV simulation. |
| `python_codes/LV_simulation/LV_simulation.py` | Top-level multiphysics driver, time loop, state updates, MPI assembly, and most output handling. |
| `python_codes/LV_simulation/mesh/mesh.py` | HDF5 mesh and fiber loading, function spaces/functions, boundary conditions, weak form, and mesh diagnostics. |
| `python_codes/LV_simulation/dependencies/forms.py` | Deformation measures, passive constituent energies/stresses, active-stress coupling, and cavity constraints. |
| `python_codes/LV_simulation/dependencies/nsolver.py` | Newton solution of the coupled finite-element equations. |
| `python_codes/LV_simulation/half_sarcomere/` | Calcium handling, myofilament kinetics, cross-bridge stress, and half-sarcomere state. |
| `python_codes/LV_simulation/circulation/` | Four- or six-compartment closed-loop circulation. |
| `python_codes/LV_simulation/heart_rate/` | Activation timing and beat-state logic. |
| `python_codes/LV_simulation/baroreflex/` | Optional baroreflex and controlled-parameter dynamics. |
| `python_codes/LV_simulation/growth/` | Optional growth/remodeling model and reference-configuration update. |
| `python_codes/LV_simulation/fiber_reorientation/` | Optional legacy fiber-reorientation subsystem; not enabled by the current base input. |
| `python_codes/LV_simulation/display/` | CSV/Excel-driven multi-panel plotting helper. |
| `python_codes/mesh_generation/` | Mesh-generation and VTK conversion utilities; these are not run by the LV driver. |
| `demos/base/sim_inputs/` | Maintained baseline, mesh, and `k_on` calibration inputs. |
| `demos/`, `demos_n/`, `python_codes_Sara/` | Historical studies, copied trees, and archived inputs; do not assume they use the maintained Fibrosis path. |
| `tests/` | Lightweight configuration and source-level constitutive/execution-path checks. |

## Environment and dependencies

The maintained solver uses legacy FEniCS/DOLFIN syntax and Python 2 language
features. No reproducible requirements file or container recipe is currently
committed, so exact compatible versions must be confirmed on the target
cluster.

Dependencies imported by the maintained path include:

- Python 2.7-compatible FEniCS/DOLFIN and UFL;
- MPI, `mpi4py`, PETSc, and `petsc4py` through the FEniCS installation;
- NumPy, SciPy, and pandas;
- HDF5 support in DOLFIN;
- Matplotlib and seaborn for `display/multi_panel.py`;
- VTK and additional scientific packages for some mesh-generation utilities.

Existing SLURM examples run the code inside a site-specific Singularity image.
Those scripts contain absolute paths, accounts, partitions, and historical
input paths and must be adapted rather than copied verbatim.

### Environment checks

Run these inside the intended FEniCS environment:

```bash
python --version
python -c "import dolfin, mpi4py, numpy, scipy, pandas"
python -c "import dolfin; print(dolfin.__version__)"
```

The repository does not establish a supported version matrix. Successful
imports alone do not validate solver compatibility.

## Running the current baseline

The driver expects exactly two command-line arguments after the script:
the mode `LV_sim` and a JSON instruction path. Relative paths in the current
input are resolved from the process working directory. The current baseline
therefore supports this layout:

```bash
cd python_codes
python MyoFE.py LV_sim ../demos/base/sim_inputs/base_instruction.json
```

For MPI execution:

```bash
cd python_codes
mpiexec -np 32 python MyoFE.py LV_sim \
  ../demos/base/sim_inputs/base_instruction.json
```

Use the Python executable and MPI launcher supplied by the same FEniCS
environment. A full production run is computationally expensive and is not
part of the documentation tests.

The four maintained `k_on` sensitivity inputs are:

- `base_instruction_kon_3e7.json`;
- `base_instruction_kon_6e7.json`;
- `base_instruction_kon_1e8.json`;
- `base_instruction_kon_1p735e8.json`.

Each has a separate output directory. The original baseline remains the
control.

## Configuration overview

JSON scalar values are generally represented as one-item lists because the
reader and model constructors access values using `[0]`. The loader uses
`json.load`, recursively converts Python 2 Unicode strings to byte strings,
and otherwise retains the parsed structure.

### Top-level sections

| Section | Purpose |
| --- | --- |
| `protocol` | Number and size of time steps plus optional activation windows and perturbations. |
| `heart_rate` | Active period, quiescent period, and first activation time. |
| `mesh` | Mesh path, user-declared function spaces, passive-law parameters, and optional solver/boundary/growth-related settings. |
| `model.circulation` | Compartment scheme, total blood volume, resistances, compliances, and slack volumes. |
| `model.half_sarcomere` | Reference lengths, membrane kinetics, myofilament kinetics, and force parameters. |
| `model.baroreflex` | Optional baroreflex model. It is constructed whenever this section exists; protocol windows determine when its controls are active. |
| `model.growth` | Optional growth/remodeling components. Absent from the current base input. |
| `model.fiber_reorientation` | Optional fiber-reorientation model. Absent from the current base input. |
| `output_handler` | Scalar CSV path, mesh/XDMF directory, requested mesh fields, spatial averaging, and output frequency. |

### Current passive Fibrosis law

`mesh.forms_parameters.passive_law_parameters.passive_law` is
`Xi-ground-collagen`. Required groups are validated before weak-form creation.
The three fractions must sum to one and are not silently normalized.

The implemented energy is

\[
\Psi = \phi_g\Psi_g + \phi_m\Psi_m + \phi_c\Psi_c + \Psi_{\mathrm{inc}},
\]

where:

- ground matrix uses an isotropic exponential function of
  \(I_1=\mathrm{tr}(C_e)\) with `a_g`, `b_g`;
- myofiber uses the retained tension-only Xi law based on
  `hsl/hsl0`, with `c2`, `c3`;
- collagen uses independent fiber, sheet, and sheet-normal exponentials with
  `a_cf/b_cf`, `a_cs/b_cs`, and `a_cn/b_cn`; each uses
  \(I_4^*=\max(I_4,1)\) through UFL `conditional`;
- incompressibility remains a mixed displacement-pressure constraint
  \(-p(J-1)\) in the current weak form.

All three passive components currently use `Cmat = Fe.T*Fe`. In the normal
baseline `Fg` is absent, so `Fe = F`. The optional growth formulation supplies
`Fg`, in which case `Fe = F inv(Fg)`.

The current base parameters are stored directly in the JSON. They should not
be described as calibrated merely because they are executable; see the
detailed review for provenance and unit caveats.

## Simulation workflow

At a high level:

1. `MyoFE.py` loads and recodes the JSON and creates `MPI.COMM_WORLD`.
2. `LV_simulation` constructs a template half-sarcomere, mesh, weak form,
   nonlinear solver, one half-sarcomere object per local quadrature point,
   circulation, heart-rate state, and optional modules.
3. `MeshClass` reads the HDF5 mesh, facet markers, `f0`, `s0`, `n0`, and local
   coordinate fields; creates CG, Real, DG, and Quadrature spaces; validates
   passive parameters; and prints rank-zero mesh/material-point diagnostics.
4. Each time step advances circulation volumes, optional controls and
   perturbations, heart-rate activation, local calcium/myofilament ODEs, and
   finite-element state fields.
5. The Newton solver solves passive mechanics, active stress, the LV cavity
   volume constraint, incompressibility, and rigid-body constraints together.
6. LV pressure, half-sarcomere lengths, passive stress feedback, model data,
   and optional growth/reorientation state are updated.
7. At `frequency_n`, scalar and spatial data are recorded and requested XDMF
   fields are written.
8. MPI-local spatial data are gathered to rank 0 for final CSV and optional
   canonical NPY output.

## Mesh, fiber, and quadrature representation

The current base mesh is read from an HDF5 group named `ellipsoidal`.
Facet boundaries and material directions are read from:

- `ellipsoidal/facetboundaries`;
- `ellipsoidal/eF` (`f0`, fiber);
- `ellipsoidal/eS` (`s0`, sheet);
- `ellipsoidal/eN` (`n0`, sheet normal).

The current material-coordinate and scalar quadrature spaces are degree-2
Quadrature elements on tetrahedra. A reported 32-rank baseline has 1,252
cells, 442 unique vertices, four quadrature points per cell, and 5,008 global
material points. On rank 0, `(456,)` for a direction vector is 456 scalar
components or 152 three-component local vectors—not 456 material points.

The diagnostic uses owned-cell counts, DOLFIN's unique global vertex count,
global function-space dimensions, and global DOF IDs to avoid counting ghost
entities twice.

## Outputs

### Time-series CSV

`output_data_path` receives a CSV assembled from circulation fields, LV state,
heart-rate state, and enabled optional-module scalar data. Output rows are
allocated according to `no_of_time_steps / frequency_n + 1`.

### Spatial CSV

- With `dumping_spatial_in_average: true`, rank-weighted spatial averages are
  written to `spatial_data.csv`.
- With averaging disabled, one `<field>_data.csv` is written per selected
  field. Rows are saved states and columns are global quadrature DOFs plus a
  `time` column.

The default spatial set contains fiber components, quadrature coordinates,
local coordinate directions, displacement sampled at quadrature points,
selected passive/active stresses, half-sarcomere length, and strains. It does
**not** include every half-sarcomere, membrane, or myofilament state even when
`save_outputs` is `all`; see Known limitations.

### Canonical NPY arrays

When spatial averaging is disabled and the required fields are available,
rank 0 writes:

- `quadrature_dof.npy`: `(global_points, 3)` reference coordinates;
- `ecc.npy`, `err.npy`, `ell.npy`: local-coordinate vectors;
- `norm_dist_endo.npy`: normalized endocardial-distance data;
- `f0_vs_time.npy`: `(global_points, 3, saved_states)` fiber history.

The writer validates global DOF coverage, coordinate ordering, shapes, finite
values, and local-basis norms/orthogonality before reporting diagnostics.

### XDMF

`mesh_output_path/solution.xdmf` receives fields listed in
`mesh_object_to_save`. The current baseline requests `displacement`,
`active_stress`, and `hs_length`. Displacement is written in its natural CG2
space; scalar quantities are projected to the configured `scalar` output
space. These XDMF outputs are the direct ParaView-oriented path.

## Post-processing and visualization

- `python_codes/LV_simulation/display/multi_panel.py` reads CSV or Excel data
  and a JSON plotting template to generate Matplotlib multi-panel figures.
- `python_codes/mesh_generation/vtk_py/` contains numerous VTK conversion,
  fiber, surface, and mesh utilities. They are heterogeneous legacy scripts,
  not one validated post-processing pipeline.
- CSV and NPY outputs can be read with pandas/NumPy. XDMF outputs are intended
  for ParaView, subject to the DOLFIN/XDMF versions used on the cluster.

## Common errors and troubleshooting

| Symptom | Check |
| --- | --- |
| `ImportError` for `dolfin`, PETSc, or MPI | Use the legacy FEniCS container/environment and ensure Python and `mpiexec` come from the same installation. |
| Mesh HDF5 cannot be opened | Run from `python_codes` for the current relative paths, confirm the mesh is a real HDF5 file, and check the `ellipsoidal` datasets. |
| Passive-law validation error | Supply every nested `ground_matrix`, `myofiber`, and `collagen` key and all three fractions; do not rely on generic Holzapfel–Ogden fallbacks. |
| Fractions error | Make `phi_g + phi_m + phi_c` approximately one. The validator intentionally does not renormalize. |
| MPI output cardinality/order error | Review `[FibrosisMeshDiagnostics]` and `GLOBAL OUTPUT DIAGNOSTICS`; do not infer global counts from rank-0 local vectors. |
| Existing output is overwritten | Give each case unique `output_data_path` and `mesh_output_path`, as in the `k_on` study inputs. |
| Newton failure | Inspect rank-zero residual output and the flushed completed states. Do not change scientific parameters merely to bypass an interface error. |

## Known limitations

- There is no pinned environment specification or automated end-to-end test.
- The maintained solver contains Python 2 syntax and is not directly runnable
  under standard Python 3.
- Several directories contain archived duplicates and study-specific cases;
  they are not guaranteed to match the current Fibrosis input schema.
- `save_outputs: all` means a maintained default subset, not every available
  internal state. Sheet and sheet-normal vector components are notably absent
  from the default spatial list, although their infrastructure exists.
- Canonical NPY output is skipped when spatial averaging is enabled.
- The current base JSON contains a baroreflex model and a protocol activation
  window at 20–40 s, while its 1,000 steps at 0.001 s cover only about 1 s.
  The baroreflex object is created but its protocol control window is not
  reached in that run.
- Some comments, optional branches, and archived documentation still use
  historical terminology or function-space names. Their presence does not
  imply activation in the current base case.
- Units are not defined by a central unit system. Pressure conversion in code
  supports Pa-to-mmHg interpretation, but geometry, volume, flow, and every
  output column require case-specific verification.
- Growth and fiber reorientation are optional, less-tested paths and retain
  legacy assumptions not exercised by the current baseline.

## Lightweight checks

The repository's current inexpensive tests can be run without launching the
production simulation:

```bash
python -m unittest \
  tests.test_fibrosis_config \
  tests.test_fibrosis_mesh_path \
  tests.test_constitutive_tangent \
  tests.test_holzapfel_ogden

for f in demos/base/sim_inputs/*.json; do
  python -m json.tool "$f" >/dev/null || exit 1
done
```

The unit tests are primarily validator, numerical formula, and source-level
regression checks. They do not assemble or solve the full FEniCS problem.
