# Opt-in bounded output

## Dependency audit

Simulation state includes the mixed FE solution, quadrature cellular state,
MyoSim populations, circulation/baroreflex state, growth/fibrosis histories,
ECM cycle integrals and fiber-reorientation state. These remain independent
of output selection. `sim_data`, `local_spatial_sim_data`,
`spatial_sim_data`, `f0_vs_time_array`, projected visualization fields and
CSV/NPY tables are output-only. The former unconditional
`f0_vs_time_array(points,3,timesteps+1)` was never populated because its
writer is commented out; it is no longer allocated.

The principal legacy costs were simulation-length pandas tables for every
spatial field on every rank, a second global table on rank zero, gathering
complete histories, output-only projections of unrequested stresses, and
periodic rewriting of all prior CSV rows.

| Data | Calculation dependency | Output treatment |
|---|---|---|
| Mixed FE solution, pressure and active tensor | weak form/nonlinear solve | never gated by output |
| MyoSim populations, HSL increments and passive feedback | cellular kinetics | never gated by output |
| Circulation, heart-rate and baroreflex state | next timestep | scalar snapshots only when selected |
| Growth signals/setpoints and constituent masses | growth/remodeling | internal histories retained; exports selected separately |
| Fibrosis cycle integrals/fractions | composition update | internal histories retained; exports selected separately |
| ECM cycle integrals/reference | homeostatic calibration | always calculated when enabled; large CSV history remains opt-in |
| Fiber-reorientation state | physical fiber update | retained only when the module is active; output history is optional |
| Projected stress/strain/visualization fields | saving and plotting only | evaluated lazily at requested save events |
| Reference coordinates and local bases | interpretation metadata | gathered and written once |

## Configuration

Legacy behavior remains the default. Opt in with:

```json
{
  "output_mode": ["buffered"],
  "output_batch_size": [100],
  "global_csv_frequency_n": [10],
  "spatial_csv_frequency_n": [10],
  "mesh_frequency_n": [100],
  "global_csv_fields": ["time", "heart_rate", "pressure_ventricle"],
  "spatial_csv_fields": ["lx", "ly", "lz", "fiber_stretch_raw"],
  "ecm_spatial_csv_fields": ["ecm_instantaneous_fiber_stress"],
  "output_profiling": [false]
}
```

Omit either field list to retain the corresponding legacy selection while
still bounding its history. Unknown requested fields fail before stepping.
`save_outputs`, `spatial_data_fields`, `ecm_spatial_csv_fields`,
`mesh_object_to_save`, and `frequency_n` remain supported. The three new
frequency keys default to `frequency_n`.

Buffered CSVs keep the historical index column, field names, global
quadrature-DOF columns, and trailing spatial `time` column. Headers are
written once and batches are append-only. Existing `data.csv` causes startup
failure, preventing accidental duplicate appends. `output_metadata.json`
records configuration, selected fields, committed row counts, profiling and
whether normal completion occurred.

## Integrity and limitations

Each spatial batch is assembled by owned global quadrature DOF. Missing,
duplicate or out-of-range ownership fails the flush. A solver `RuntimeError`
flushes completed rows and leaves `complete: false` before re-raising.
Repeated empty flushes are no-ops.

Pure-Python regression fixtures confirm identical deterministic row values,
timestamps and ordering for frequencies 1 and 10, non-divisible final
batches, repeated flushes and early partial flushes. They cannot establish
organ-model numerical equivalence: DOLFIN/MyoSim/growth/fibrosis comparisons
remain a required short-container validation before production use. Buffered
mode is therefore opt-in and legacy mode remains the default.

Static `quadrature_dof.npy`, `ecc.npy`, `err.npy`, `ell.npy`,
`norm_dist_endo.npy`, and global DOF IDs are written once at startup;
coordinate CSVs contain one immutable reference row. Full `f0_vs_time.npy`
is not emitted by the minimal profile. Set `save_f0_history` and request
`f01`, `f02`, and `f03` to use the disk-backed NPY compatibility path without
retaining its full history in RAM.

The optimized ECM case is a scientifically minimal profile. A detailed-debug
profile can add `collagen_passive`, `bulk_passive`, `total_stress_spatial`,
`Ell`, `Err`, `Ecc`, `dx`, `dy`, and `dz`; this increases I/O but never gates
their underlying physical calculations.

## Performance interpretation

`output_profiling` records MyoSim update, FE solve, extraction, buffer
preparation, MPI transfer, CSV writing, XDMF writing and total wall-time
categories in metadata. The maximum accumulated rank time is reported.
Profiling is disabled by default.

`validation/benchmark_buffered_output.py` reports an analytical float64
payload comparison and process RSS for a reproducible synthetic case. It is
not a substitute for production MPI timing; only the short cluster
validation can establish end-to-end speedup and numerical equivalence.

For 2,500 points, 13 spatial fields and 4,201 saved rows, the committed
benchmark defaults estimate 1,092,260,000 bytes of legacy float64 payload
versus 26,000,000 bytes for a 100-row buffer: a 42.01-fold numeric-payload
reduction. Pandas object overhead and MPI temporary storage are additional,
so this is a conservative payload comparison rather than measured production
peak RSS.
