# Br baseline and constant-volume AS fibrosis configurations

## Authoritative inputs

The maintained baseline is now
`demos/base/sim_inputs/base_instruction_Br.json`. The two derived cases are:

- `base_instruction_Br_fibrosis_control.json`;
- `base_instruction_Br_fibrosis_AS.json`.

All three use Xi-ground-collagen with `(phi_m, phi_g, phi_c) = (0.700,
0.274, 0.026)`, the calibrated coefficients and MyoSim values specified for
the Br case, the requested HCM mesh, 60,000 steps of 0.001 s, baroreflex
activation from 5--60 s, and the remodeling schedule from 20--60 s.

`protocol.growth` is deliberately used only as the fibrosis scheduling window.
There is no `model.growth` block in either fibrosis case. At run startup the
fibrosis module consumes exactly one protocol growth activation and refuses to
coexist with geometric growth. Therefore the geometric growth solver and
`ALE.move` are not invoked.

## Case differences from the Br baseline

The healthy fibrosis control adds only:

- `model.fibrosis` with the preliminary sensitivity values `tau_c=50`,
  `phi_c_max=0.15`, `stress_epsilon=1.0`, `gamma_c=1.5`, three homeostatic cycles, and four
  cycles per remodeling update;
- a dedicated absolute CSV output path;
- a dedicated absolute mesh-output path;
- fibrosis/stress/composition output fields.

The AS fibrosis case has the same differences plus exactly one perturbation:

- `protocol.perturbation.perturbations[0].level`: absent -> `circulation`;
- `.variable`: absent -> `aorta_resistance`;
- `.t_start_s`: absent -> `30`;
- `.t_stop_s`: absent -> `40`;
- `.total_change`: absent -> `170`.

The initial aortic resistance remains 20; the circulation perturbation code
ramps the total `+170` change over the specified interval.

## Output-field compatibility

The requested mechanical fields that are valid without geometric growth are
retained: displacement, active stress, total stress, total passive stress,
myofiber passive stress, collagen passive stress, half-sarcomere length,
DG0 active stress, total-stress vector, transverse/helical angles, constituent
fractions/densities, and fibrosis stress/stimulus fields.

The following requested names are **not** placed in `mesh_object_to_save`:

- `local_theta_vis_*`, `global_theta_vis_*`;
- `deviation_*`, `stimulus_*`, `setpoint_*` from the geometric-growth module.

`MeshClass` creates those Functions only when `model.growth` exists. Including
them in a fibrosis-only case would either raise a missing-key error or, due to
legacy output fall-through, write a stale Function under a misleading name.
Enabling geometric growth merely to create these outputs would violate the
constant-volume experiment. Their meaningful fibrosis replacements are
`fibrosis_cycle_averaged_total_fiber_stress`,
`fibrosis_homeostatic_fiber_stress`, and
`fibrosis_normalized_stimulus`.

## Nonlinear fibrosis stimulus

The configured cases retain the raw normalized stimulus `G_c` and use
`G_c_effective = G_c**gamma_c` only in the collagen evolution rate. The exact
bounded update is

`phi_c_new = phi_c_old + (phi_c_max - phi_c_old) *`
`(1 - exp(-G_c_effective * dt / tau_c))`.

`gamma_c` defaults to `1.0` in older inputs. The Br control and AS cases use
the exploratory value `1.5`. Thus values below one are attenuated, values
above one are amplified, and zero remains zero. There is no stimulus clipping
or additional scaling. Both raw and effective local stimuli and their global
minimum/mean/maximum are output so that high maxima can be checked against
small homeostatic-stress denominators. No result files are stored in this
repository, so the typical stimulus range cannot be established statically;
it must be measured in the control and AS cluster runs before calibration.

An exploratory sensitivity study should copy the AS input and vary only
`gamma_c` over `0.75`, `1.0`, `1.5`, and `2.0`, holding `tau_c`, `phi_c_max`,
loading, circulation, and material parameters fixed. A mean collagen fraction
near 0.12 is a comparison target, not an enforced endpoint; it must not be
confused with the maximum local fraction or directly equated with histology
without validating the common-density mixture interpretation.

## Output paths

Healthy control:

- `/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test_control/sim_output/data.csv`
- `/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test_control/mesh_output`

AS fibrosis:

- `/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test/sim_output/data.csv`
- `/mnt/gpfs2_4m/scratch/sba431/demos/Fibrosis/AS_test/mesh_output`

## Running

From the repository's `python_codes` directory:

```bash
mpiexec -np 32 python MyoFE.py LV_sim \
  ../demos/base/sim_inputs/base_instruction_Br_fibrosis_AS.json
```

The mesh path is valid from that working directory: `../demos/HCM_paper/...`
resolves inside this repository. Full serial/MPI FEniCS validation remains
required before interpreting physiological results.

For the complete 32-rank control/AS comparison plus invariant checks and plots:

```bash
export FENICS_IMAGE=/path/to/working-fenics-image.sif
sbatch validation/run_br_as_fibrosis.slurm
```

`postprocessing/plot_spatial_collagen.py` reads the existing pointwise spatial
CSV contract and creates a before/after X-Z collagen map. ParaView should be
used with `solution.xdmf` for full 3-D collagen and passive-stress inspection.
The `collagen_passive` XDMF field is the fiber-direction collagen contribution;
`total_passive` is the complete passive fiber-direction response.
