# Finer-mesh baseline LV setup

- Mesh file: `demos/finer_mesh/ellipsoidal.hdf5`
- Baseline JSON: `demos/finer_mesh/baseline_15_cycles_no_baroreflex_no_disarray.json`

## Mesh generation source
This setup uses the repository's finer ellipsoidal mesh artifact produced by the mesh-generation workflow under:
`python_codes_org/mesh_generation/output_files/HCM_paper/finer_100%/ellipsoidal.hdf5`.

The file is placed at `demos/finer_mesh/ellipsoidal.hdf5` for this demo.

## Run command
From repository root:

```bash
python MyoFE.py LV_sim demos/finer_mesh/baseline_15_cycles_no_baroreflex_no_disarray.json
```

or:

```bash
./demos/finer_mesh/run_baseline_15_cycles.sh
```

## Smoke test
Create a temporary copy and reduce `protocol.no_of_time_steps` to `10`, then run the same command against that temporary JSON.
