# Aortic-stenosis-driven, volume-neutral fibrosis

## Status and provenance

Related published growth framework: Sharifi et al. (2024), *Multiscale Finite
Element Modeling of Left Ventricular Growth in Simulations of Valve Disease*,
doi:10.1007/s10439-024-03497-x. The fibrosis kinetics below are a new project
extension and are not presented as part of that publication.

This is an optional first sensitivity model, not a calibrated biological law.
It reuses the loading and mechanical stimulus present in
`demos/growth_AS/sim_inputs/input_parameters.json`: the aortic resistance is
increased by 150 model resistance units from 30 to 40 s, total fiber-direction
stress is evaluated as `f0 . total_stress . f0`, and remodeling is evaluated
every four cardiac cycles. The historical AS example uses this stress to drive
sheet and sheet-normal geometric growth. The new extension instead exchanges
ground matrix for collagen and never calls the growth solver or `ALE.move`.

The validation inputs are derived from the current maintained Fibrosis
baseline, not from the historical example's obsolete `semi_structural`
material block. Thus the current Xi-ground-collagen coefficients, initial
fractions `(0.70, 0.27, 0.03)`, MyoSim parameters, circulation compartments,
and mesh path remain unchanged. Only the published-example AS resistance
perturbation and timing are reused.

## Mechanical stimulus

At every mechanics step, the existing total-stress tensor is projected using
the same DG-to-Quadrature route as `growth_component.return_signal()`:

`S_ff,total = f0 . total_stress . f0`.

The code integrates this field in time and divides by the actual accumulated
cycle duration. Fibrosis uses `protocol.growth.activations` as its remodeling window; the maintained Br cases activate it at 20 s, matching the historical growth
activation. Three cycle averages establish the local healthy homeostatic
field before the AS perturbation starts at 30 s. Each remodeling update then
uses the duration-weighted average over four completed cycles. The positive
normalized stimulus is

`G_c = max((S_c - S_homeo) / max(abs(S_homeo), S_epsilon), 0)`.

`S_epsilon` has the same units as the model stress. It prevents division by a
near-zero setpoint; its example value requires sensitivity testing.

## Collagen update and composition exchange

For the actual accumulated remodeling interval `Delta t`, the exact update is

`phi_c(new) = phi_c + (phi_c_max-phi_c)*(1-exp(-G_c*Delta t/tau_c))`.

It is monotone and bounded. Zero or negative stimulus leaves collagen
unchanged. The validation values `tau_c=50 s` and `phi_c_max=0.15` are clearly
labeled accelerated sensitivity values, not physiological calibration.
`phi_c_max` is validated against the configured initial collagen and ground-matrix fractions, so the requested exchange cannot consume more ground matrix than is available.

The target collagen fraction is converted to a referential-density increment. The example common density is `1.0`, so reported masses are normalized density-times-mesh-volume quantities unless the user supplies a verified physical density and mesh unit.
Exactly the same density is removed from ground matrix, myofiber density is
unchanged, and total referential density is checked pointwise and globally.
Fractions are then recalculated from the densities and written into the same
Quadrature Functions read by `Forms`. Consequently subsequent passive energy,
stress, residual, and tangent evaluations use

`W = phi_m W_Xi + phi_g W_g + phi_c W_c`

without changing any constituent law or coefficient.

This is phenomenological constituent exchange at common density. It is not
positive-growth deposition, degradation, prestretch, fibroblast signaling, or
replacement fibrosis.

## Why reference myocardial volume is preserved

The fibrosis configurations contain no `model.growth`, require
`geometric_growth_enabled=false`, and the module refuses to initialize beside
the geometric growth module. Therefore `Fg=I`, `Jg=1`, no growth mechanics
solve occurs, and no `ALE.move` changes the reference mesh. Existing elastic
incompressibility remains untouched. Reference myocardial volume is assembled
from the mesh after every fibrosis event and compared with its initial value.
LV cavity volume remains free to vary during each cardiac cycle.

## Validation cases

- `validation/configs/as_fibrosis_healthy.json`: initial aortic resistance 20;
  no resistance perturbation.
- `validation/configs/as_fibrosis_aortic_stenosis.json`: identical model plus
  the existing `+150` aortic-resistance perturbation from 30--40 s.

Both use identical fibrosis sensitivity parameters and isolated output paths.
They preserve the current baseline passive, active, circulation, and mesh
inputs. They run for 150 s, matching the duration of the historical AS input.

Run on the legacy cluster environment:

```bash
export FENICS_IMAGE=/path/to/working-fenics-image.sif
sbatch validation/run_as_fibrosis_validation.slurm
```

The job creates time-series CSV, rank-local spatial CSV/NPY data according to
the existing output handler, `solution.xdmf` fields for ParaView, and:

- `validation/results/as_fibrosis/plots/fibrosis_evolution.png`;
- `validation/results/as_fibrosis/plots/fibrosis_pv_loops.png`.

The evolution plot compares mean stress, collagen fraction, ground-matrix
fraction, and reference myocardial volume. The PV plot selects complete beats
from the earliest and latest fibrosis-event states. Spatial XDMF fields include
all three fractions, all constituent densities, total density, cycle-averaged
stress, homeostatic stress, normalized stimulus, total/passive stress, and
active stress.

## Pass criteria

1. `fib_reference_volume_residual` remains at numerical zero.
2. `fib_total_mass_residual` remains at numerical zero.
3. `fib_phi_m_min/mean/max` remain 0.70.
4. At every point `phi_m+phi_g+phi_c=1`, with nonnegative ground matrix.
5. Healthy control stimulus remains near zero after stabilization and collagen
   remains near 0.03. This is a runtime criterion, not guaranteed by design if
   the early baseline remains transient.
6. AS produces positive stimulus where total fiber stress exceeds the local
   homeostatic field and increases collagen only there.
7. Re-running mechanics after an event changes passive response under
   collagen-recruiting deformation.
8. Serial and MPI runs agree in global mass and fraction summaries.

## Passive pressure-volume validation

The dynamic run supplies before/after cardiac PV loops, not an EDPVR. A true
passive pressure-volume curve requires a separate passive-inflation calculation
with active stress disabled. The existing `validation/passive_lv_inflation.py`
does not yet import a spatially evolved collagen snapshot, so it cannot yet
produce the requested post-fibrosis passive curve without an additional,
validated state-transfer feature. This limitation is explicit: do not call the
filling limb of the dynamic loop an EDPVR. The next validation task is to save
and reload the distributed fraction fields into the passive-inflation driver.

## Uncalibrated assumptions

The parameters `tau_c`, `phi_c_max`, and `S_epsilon`; the use of the first three
post-activation cycles as homeostasis; common constituent density; direct
matrix-to-collagen exchange; absence of turnover/prestretch; and stress-only
activation all require scientific calibration or sensitivity analysis. The
Sharifi--Wenk geometric growth formulation is preserved separately; this
fibrosis equation is a new proposed extension and is not attributed to that
publication.
