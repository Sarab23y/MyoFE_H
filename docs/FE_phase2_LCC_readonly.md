# Read-only LCC evidence collection — no simulation or mapping submission

These commands inspect originals and create a separate evidence directory.
They do not repair files, pull/reset checkouts, change parameters, seal reports,
or launch FE/MyoSim. Run after confirming the baseline job is stopped. Hashing
large files can take minutes of storage I/O; use an appropriate site-approved
interactive allocation if required by LCC policy.

## 1. Locate the run and establish current provenance

```bash
export RUN=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_60s_healthy_all_outputs_stride1
export PROD=/home/sba431/MyoFE_H
export DIAG=/home/sba431/MyoFE_H_diagnostics
export IMAGE=/home/sba431/fenics.img
export SCRATCH=/mnt/gpfs2_4m/scratch/sba431/demos/ECM
test -d "$RUN"
export AUDIT=$(mktemp -d "$SCRATCH/fe_readonly_audit_XXXXXX")
echo "$AUDIT"

{
  date -u
  hostname
  git -C "$PROD" rev-parse HEAD
  git -C "$PROD" status --short
  git -C "$PROD" diff --stat
  git -C "$DIAG" rev-parse HEAD
  git -C "$DIAG" status --short
  stat -c '%n size=%s modified=%y' "$RUN/instruction.json" "$RUN/sim_output/output_metadata.json"
  sha256sum "$RUN/instruction.json" "$IMAGE"
  cat "$RUN/sim_output/output_metadata.json"
} > "$AUDIT/current_provenance.txt" 2>&1

sha256sum \
  "$PROD/python_codes/MyoFE.py" \
  "$PROD/python_codes/LV_simulation/LV_simulation.py" \
  "$PROD/python_codes/LV_simulation/mesh/mesh.py" \
  "$PROD/python_codes/LV_simulation/dependencies/forms.py" \
  "$PROD/python_codes/LV_simulation/dependencies/nsolver.py" \
  "$PROD/python_codes/LV_simulation/dependencies/fenics_2017_compat.py" \
  "$PROD/python_codes/LV_simulation/half_sarcomere/half_sarcomere.py" \
  "$PROD/python_codes/LV_simulation/half_sarcomere/myofilaments/myofilaments.py" \
  "$PROD/python_codes/LV_simulation/half_sarcomere/myofilaments/move.py" \
  "$PROD/python_codes/LV_simulation/half_sarcomere/myofilaments/kinetics.py" \
  > "$AUDIT/current_source_sha256.txt"

singularity exec --cleanenv "$IMAGE" python - <<'PY' > "$AUDIT/runtime_and_defaults.txt" 2>&1
import sys, dolfin
print(sys.version)
print(dolfin.__version__)
print(dolfin.NonlinearVariationalSolver.default_parameters())
print(dolfin.NewtonSolver.default_parameters())
PY
```

Current Git HEAD/defaults/hashes do **not** establish what was imported at launch.
Recover the submitted batch script, launch-time source snapshot and dirty diff,
actual baseline job ID, scheduler accounting and its stdout/stderr. Preserve
all originals. If no historical snapshot exists, state that explicitly.

Inspect the submitted input's mesh path, solver block, kinetic scheme/bin
range/compliance factor, filament lengths and optional health diagnostics.
Do not substitute a repository configuration. A copied instruction is useful
but must be checked against the argument actually read by the submitted job.

## 2. Compare original files with the Windows copy

```bash
stat -c '%n size=%s modified=%y' \
  "$RUN/mesh_output/solution.xdmf" "$RUN/mesh_output/solution.h5" \
  "$RUN/mesh_output/mesh_with_mpi.xdmf" "$RUN/mesh_output/mesh_with_mpi.h5" \
  > "$AUDIT/original_file_sizes.txt"

sha256sum "$RUN/mesh_output/solution.xdmf" "$RUN/mesh_output/solution.h5" \
  "$RUN/mesh_output/mesh_with_mpi.xdmf" "$RUN/mesh_output/mesh_with_mpi.h5" \
  > "$AUDIT/original_mesh_sha256.txt"

singularity exec --cleanenv "$IMAGE" python - "$RUN/mesh_output" \
  > "$AUDIT/original_mesh_open_checks.txt" 2>&1 <<'PY'
from __future__ import print_function
import os, sys, xml.etree.ElementTree as ET
import h5py
root = sys.argv[1]
for name in ('solution.xdmf', 'mesh_with_mpi.xdmf'):
    try:
        count = 0
        for event, element in ET.iterparse(os.path.join(root, name), events=('end',)):
            count += 1
            element.clear()
        print(name, 'XML PASS', count, 'elements')
    except Exception as error:
        print(name, 'XML FAIL', repr(error))
for name in ('solution.h5', 'mesh_with_mpi.h5'):
    try:
        with h5py.File(os.path.join(root, name), 'r') as handle:
            print(name, 'HDF5 OPEN PASS', list(handle.keys()))
    except Exception as error:
        print(name, 'HDF5 OPEN FAIL', repr(error))
PY
```

Windows observations, for exact comparison:

| File | Bytes | SHA256 |
|---|---:|---|
| solution.xdmf | 104857600 | c4aeaf606b032c6fcdf8bd6e8fecb3ebc00729efd3915cc3d6a39a9d94a6d2a9 |
| solution.h5 | 5696314416 | 8305bd6ad2eab3c8fce73be35278812400ea1230bcc3af09f916263224e60413 |
| mesh_with_mpi.xdmf | 785 | 4f762b0afde08ab2f2d3cdeb74e05ee6ceb6cd9ab57dfc04d67a23b0ea63ff8e |
| mesh_with_mpi.h5 | 75704 | ee8d5d7a79ecc096e7823e48f3a658880eeb68e949a1292ac93f17bef253a706 |

Repeat stat afterward if any producer/copy process may still be writing. An
identical hash establishes identical bytes at comparison time, not their
historical origin. An openable HDF5 root does not guarantee every dataset is
complete. Inspect time coverage, dimensions and referenced datasets before
using any displacement. An intact visualization file still may lack CG2 edge
coefficients; look for actual Function-space checkpoint data.

## 3. Inspect existing mapping status before considering a rerun

```bash
sacct -j 6340763 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
# If a subsequent diagnostic job exists, set its actual ID explicitly:
if test -n "${MAPPING_JOB:-}"; then
  sacct -j "$MAPPING_JOB" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
  squeue -j "$MAPPING_JOB"
fi
find "$SCRATCH" -maxdepth 1 -type d -name 'phase_mapping_*' -print
```

For the actual latest directory found, set MAPPING to its exact full path, then:

```bash
: "${MAPPING:?Set MAPPING to the actual latest diagnostic report directory}"
ls -l "$MAPPING"
test ! -f "$MAPPING/failure.txt" || cat "$MAPPING/failure.txt"
test ! -f "$MAPPING/environment.txt" || tail -n 100 "$MAPPING/environment.txt"
test ! -f "$MAPPING/preflight/preflight.json" || cat "$MAPPING/preflight/preflight.json"
test ! -f "$MAPPING/mapping_manifest.json" || cat "$MAPPING/mapping_manifest.json"
```

Require all ten existing reports: communicator 1/32, scalar 1/32, mixed 1/32,
geometry 1/32 and the scalar/geometry comparisons. Check actual input/mesh
hashes, rank reports, owned/scatter verification, saved coordinate/wall matching
and sealed manifest. Missing, failed or stale reports are not PASS. A source
hash mismatch after diagnostic development is a provenance mismatch, not proof
that physical mapping failed. Do not overwrite or reseal old reports.

There is deliberately no `sbatch` command here. First establish whether a
later job already ran and recover its complete evidence.

## 4. Locate authoritative state and solver histories

```bash
find "$RUN" -maxdepth 4 -type f \
  \( -iname '*checkpoint*' -o -iname '*restart*' -o -name '*.hdf5' \
     -o -name '*.h5' -o -name '*.out' -o -name '*.err' \
     -o -name '*mechanical_health*' \) -print > "$AUDIT/state_inventory.txt"

# Use the exact original baseline log, not an older healthy-optimized run:
if test -n "${BASELINE_LOG:-}"; then
  rg -n 'Newton iteration|Newton solver finished|Failed Convergence|RuntimeError|LU solver|mumps|Sim time|NEW TIME STEP' \
    "$BASELINE_LOG" > "$AUDIT/solver_log_index.txt"
fi
```

Keep each Newton history associated with its original timestep header. The
code prints the pre-increment clock; residuals alone do not give physical
admissibility or correction magnitude. Do not interpret an older-run log as
the current event history. Full restart requires much more than displacement.
