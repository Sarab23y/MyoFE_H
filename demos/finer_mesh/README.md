# Finer mesh baseline setup

## 1) Generate the mesh (required)
Run from repository root:

```bash
cd python_codes/mesh_generation
python2 generate_finer_mesh_demo.py
```

This generates:

`demos/finer_mesh/ellipsoidal.hdf5`

using the mesh-generation workflow in `python_codes/mesh_generation` and validates that DOLFIN can read dataset `ellipsoidal`.

## 2) Path checks from launch directory
If launching from `python_codes`, the JSON path resolves to:

```bash
cd python_codes
realpath ../demos/finer_mesh/ellipsoidal.hdf5
ls -lh ../demos/finer_mesh/ellipsoidal.hdf5
head -c 8 ../demos/finer_mesh/ellipsoidal.hdf5 | od -An -tx1
```

Expected HDF5 signature bytes:

`89 48 44 46 0d 0a 1a 0a`

## 3) Validate mesh readable by DOLFIN
From `python_codes`:

```bash
python2 ../demos/finer_mesh/validate_finer_mesh.py
```

## 4) Run baseline simulation
From repository root:

```bash
./demos/finer_mesh/run_baseline_15_cycles.sh
```
