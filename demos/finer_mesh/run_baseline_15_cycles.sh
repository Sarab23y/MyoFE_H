#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

MESH="demos/finer_mesh/ellipsoidal.hdf5"
SIG=$(python - <<'PY'
with open('demos/finer_mesh/ellipsoidal.hdf5','rb') as f:
    print(f.read(8).hex())
PY
)
if [[ "$SIG" != "894844460d0a1a0a" ]]; then
  echo "[finer_mesh] Mesh is not a valid HDF5 file (signature=$SIG). Regenerating..."
  (cd python_codes/mesh_generation && python2 generate_finer_mesh_demo.py)
fi

python2 python_codes_org/MyoFE.py LV_sim demos/finer_mesh/baseline_15_cycles_no_baroreflex_no_disarray.json
