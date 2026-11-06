# 60-second healthy all-output baseline

The production input is
`demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json`.
It is a fresh 60,000-step run (`dt = 0.001 s`) derived from the successful
40-second healthy case. It has no perturbations, growth, fibrosis evolution,
or fiber reorientation. Baroreflex activation remains at 5 seconds and ends
with the run; ECM calibration remains at 40 seconds with its original 5%
stability criterion and `warn` policy.

## Output contract

All three effective strides (`global_csv_frequency_n`,
`spatial_csv_frequency_n`, and `mesh_frequency_n`) and the legacy fallback
`frequency_n` are one. Buffered CSV batches contain 50 states, so the spatial
history is bounded in memory and the final partial batch is flushed normally.
The expected convention is the initial state plus 60,000 advanced states:
60,001 rows ending near 60 seconds.

`save_outputs: all` records all scalar entries exposed by each material
point's half-sarcomere, myofilament, and membrane dictionaries. In this model
that includes `Ca_cytosol`, `Ca_SR`, `J_release`, `J_uptake`, actual `k_act`,
`k_serca`, `k_1`, and `k_on`, cross-bridge states/fluxes, HSL, and stresses.
The actual rates are spatial files and are distinct from the per-point
`membranes_*_rc_*` and `myofilaments_*_rc_*` baroreflex control signals in
`data.csv`. Global output also retains the baroreflex setpoint/state,
activation flag, heart-rate counters, circulation pressures/volumes/flows,
resistance, and compliance values.

The spatial mechanics profile distinguishes `active_stress`, Xi passive
stress (`Sff_mesh` and `myofiber_passive`), `total_passive`, constraint stress,
raw/tension-only stretch, strain, displacement components, and the three ECM
diagnostic fields. Canonical `quadrature_global_dof.npy`, coordinates, wall
depth, and local bases are written once. Mechanical-health event details
remain in their diagnostic JSON output rather than being invented as dense
per-point CSV fields.

The simple two-compartment calcium model exposes release and uptake but no
separately named additional calcium fluxes. Quantities not present in its
runtime `membranes.data` dictionary cannot be recorded by this profile.

## Timestamp semantics

The driver records the initialized state at time zero. For each later row,
MyoSim, mechanics, circulation, and diagnostic updates are completed for the
current step before output; the driver's time is then advanced by `dt` at the
end of the step. The existing coupling/update order is unchanged. These rows
therefore retain the historical MyoFE timestamp convention rather than
silently relabeling staggered variables.

## LCC workflow

From `/home/sba431/MyoFE_H`:

```bash
git pull
SMOKE_JOB=$(sbatch --parsable demos/ECM/run_ecm_60s_healthy_all_outputs_stride1_smoke.slurm)
echo "$SMOKE_JOB"
squeue -j "$SMOKE_JOB"
# After the smoke job exits successfully and stride1_validation.json says PASS:
FULL_JOB=$(sbatch --parsable demos/ECM/run_ecm_60s_healthy_all_outputs_stride1.slurm)
echo "$FULL_JOB"
squeue -j "$FULL_JOB"
```

The full launcher refuses to overwrite
`/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1`,
prints a conservative storage estimate, and aborts rather than downsampling
if free space is insufficient. The smoke directory is job-specific.
