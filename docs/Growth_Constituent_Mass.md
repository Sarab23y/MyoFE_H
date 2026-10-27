# Growth-dependent constituent mass: implementation and validation

## Scope and unchanged baseline

The baseline mixture remains
`W = phi_m W_Xi + phi_g W_g + phi_c W_c`, initially `(0.70, 0.27,
0.03)`. The Xi, ground-matrix, collagen, active MyoSim, and circulation laws
are not altered by this bookkeeping. This feature only allocates positive
incremental growth mass and updates the three mixture weights.

## Quantities and density assumption

At each scalar Quadrature material point the implementation stores
referential constituent densities `rho_i` (mass per volume of the **current
reference mesh**), not arbitrary point masses. Their sum is the local tissue
referential density. Mass fractions are `w_i=rho_i/sum(rho_j)`. With the
explicit common true-density assumption, these are also the volume fractions
`phi_i` consumed by `Forms`. Different constituent densities would require a
mass-to-volume conversion that is not implemented.

For an event, `Jg=det(Fg)` describes prescribed, unconstrained growth and
`Jale=det(F_ALE)` describes the compatible displacement that is committed by
`ALE.move`. On the old reference:

`Delta rho_i = rho * max(Jg - 1, 0) * a_i`.

After moving the reference mesh:

`rho_i(new) = (rho_i(old) + Delta rho_i) / Jale`.

Thus `rho_i(new) dV(new) = (rho_i(old)+Delta rho_i) dV(old)` under the ALE
map. Prescribed growth volume and actual post-ALE geometric volume are logged
separately and are not assumed equal. All organ totals use DOLFIN `assemble`
over the physical mesh and therefore include quadrature weights and MPI
reduction; local arrays are never summed as if every point had unit volume.
Density units are user-selected through `density`; mass units are density
units times mesh-coordinate volume units. The repository does not establish a
physical density or mesh length unit, so the examples use `density=1.0` and
report consistent normalized mass units.

## Persistent material-point identity

The persistent identity is `(global cell index, cell-local scalar Quadrature
DOF position)`. Owned scalar Quadrature DOFs are identified from the DofMap
ownership range and local-to-global map. Fractions and densities are remapped
by these keys after rebuilding the primary and growth mechanics objects.
Missing, duplicate, or changed key sets raise an error. This is stronger than
checking array length and excludes ghost DOFs from owned arrays. It assumes
`ALE.move` preserves cell topology/global cell numbering, as legacy DOLFIN is
expected to do. The supplied runtime experiments must verify this assumption
in serial and MPI before scientific use.

## Coupling and timing

`apply_constituent_fractions` writes synchronized Quadrature Functions into
`passive_params['phi_m'/'phi_g'/'phi_c']` in both rebuilt mechanics objects.
`Forms._weighted_passive_energy` and its differentiated stresses then use
these Functions in every subsequent residual/Jacobian assembly. The mechanics
solve that determines the current ALE growth map uses the pre-deposition
composition; deposition affects subsequent mechanics. This is the implemented
operator split.

Collagen enrichment does not mathematically guarantee greater stress for all
deformations: increasing `phi_c` reduces the other weights, and collagen is
tension-only until its directional invariant exceeds one. Compare compositions
at an identical collagen-recruiting deformation using the material validation
tool before interpreting organ stiffening.

## Outputs

Spatial CSV output records `gr_phi_*`, constituent referential densities,
total density, deposited density, `Jg`, and `Jale`. XDMF output can record
`phi_m`, `phi_g`, `phi_c`, `constituent_density_*`, `growth_volume_ratio`, and
`growth_geometric_volume_ratio`. The ordinary time-series CSV records event
number, pre/post myocardial reference volumes, prescribed positive growth
volume, global constituent masses, and total mass-balance residual. Values are
piecewise constant between growth events.

## Controlled configurations

Three files are derived from the current baseline without changing its
material, active, circulation, mesh path, or solver parameters:

- `validation/configs/growth_mass_validation_no_growth.json`: theta=(1,1,1), proportional allocation.
- `validation/configs/growth_mass_validation_proportional.json`: theta=(1.02,1.02,1.02), allocation=(0.70,0.27,0.03).
- `validation/configs/growth_mass_validation_collagen_enriched.json`: the same growth, allocation=(0.20,0.10,0.70).

Only validation protocol duration/growth activation, required internal growth
spaces, dedicated output paths, explicit validation theta increments, and
bookkeeping/output fields differ. `prescribed_growth_increment` is a validation
control that overrides stimulus-derived increments at an event; it does not
change normal runs when absent.

Expected checks: no-growth preserves geometry, masses, fractions, and passive
response; proportional deposition increases mass while preserving fractions;
collagen-enriched deposition increases collagen mass/fraction. In all cases,
`new mass - old mass - integrated deposition` must be within configured
relative tolerance. Compare baseline and enriched material-point stresses at
the same collagen-recruiting stretch to isolate composition from geometry.

## Cluster validation

Use the known working legacy image and submit:

```bash
export FENICS_IMAGE=/path/to/working-fenics-image.sif
sbatch validation/run_growth_mass_validation.slurm
```

The job runs each case serially. Repeat the proportional case with two ranks
and compare event rows:

```bash
mpiexec -np 2 singularity exec --cleanenv "$FENICS_IMAGE" bash -lc \
  "cd $PWD/python_codes && python MyoFE.py LV_sim ../validation/configs/growth_mass_validation_proportional.json"
```

Pass criteria are: no exceptions from key mapping/synchronization; fraction
sum error <1e-8 locally; no-growth fraction/mass/volume changes within solver
tolerance; proportional fractions stay at baseline; enriched collagen
fraction rises; total and constituent residuals satisfy the configured
`1e-6` relative tolerance; serial and MPI global totals agree at matching
events. A full legacy DOLFIN run was not available in the development
container, so these FE/MPI criteria remain unverified here.

## Limitations

There is no turnover, degradation, deposition prestretch, constituent-specific
density, replacement fibrosis, active loss, or stimulus-driven collagen law.
Negative growth does not remove mass. The bookkeeping does not assert that
prescribed `Jg` equals actual geometric volume change. Establishing serial/MPI
runtime equivalence is the next milestone before stimulus-driven fibrosis.

After the three serial runs, the job invokes
`validation/check_growth_mass_results.py`; it fails if an event is absent, a
mass residual is too large, the no-growth reference volume changes, the
proportional global collagen fraction differs from 0.03, or enriched
deposition does not increase that fraction.
