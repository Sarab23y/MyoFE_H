# Fiber Disarray: Comprehensive Change Documentation

## 1) Purpose and scope
This document explains the full set of repository changes made to support **static fiber disarray** in LV simulations while disabling stress-driven runtime fiber remodeling.

The implemented intent is:
- Fibers are initialized once (aligned or disarrayed) during mesh/material setup.
- Fiber directions are **not** updated during time stepping.
- Existing constitutive and solver physics remain intact, using the initialized `f0` field.
- MPI startup/shutdown and diagnostics are hardened for PETSc + mpi4py runtime environments.

---

## 2) High-level design changes

### 2.1 Runtime reorientation disabled in simulation driver
In `python_codes/LV_simulation/LV_simulation.py`:
- Import of runtime reorientation module was removed from active wiring.
- `self.fr` is now forced to `[]` with an informational message when `fiber_reorientation` appears in input.
- Perturbation block for `fiber_reorientation` is guarded with `if self.fr:` so no invalid dereference occurs when reorientation is disabled.

**Outcome:** no stress-driven/time-step fiber update path is active in production run loop.

### 2.2 Static disarray moved to mesh initialization
In `python_codes/LV_simulation/mesh/mesh.py`:
- `initialize_functions(...)` now calls `apply_static_fiber_architecture(mesh_struct, f0, s0, n0)` exactly once.
- Static architecture supports:
  - `fiber_architecture`: `aligned` or `disarray`
  - `disarray_width`: float
  - `disarray_seed`: optional integer

**Outcome:** fiber field selection/randomization is now a startup concern, not a dynamic remodeling concern.

---

## 3) Detailed implementation in `mesh.py`

### 3.1 `apply_static_fiber_architecture(...)`
Core behaviors:
1. Reads architecture config from `mesh_struct` with defaults:
   - `aligned`, `width=0.0`, `seed=None`.
2. `aligned` mode:
   - keeps imported aligned fiber field unchanged,
   - logs misalignment stats.
3. invalid architecture:
   - falls back to aligned with warning.
4. `disarray` mode:
   - for each local quadrature point, samples
     - `v_x ~ N(1.0, width)`
     - `v_y ~ N(0.0, width)`
     - `v_z ~ N(0.0, width)`
   - normalizes vector with epsilon guard,
   - writes back to local `f0` vector and ghost-updates.
5. `disarray_width == 0.0`:
   - explicitly preserves aligned baseline and reports max absolute deviation.

Safety:
- epsilon safeguards against zero norms,
- NaN detection with runtime error.

### 3.2 `rebuild_local_coordinate_system_once(...)`
After `f0` initialization, local basis is rebuilt once:
- primary ref axis: `[0, 0, 1]`
- fallback ref: `[0, 1, 0]` when `|dot(f0, ref)| > 0.95`
- computes:
  - `s0 = normalize(cross(f0, ref))`
  - `n0 = normalize(cross(f0, s0))`
- includes epsilon fallback paths and NaN checks,
- pushes back to FEniCS vectors + ghost updates.

### 3.3 Diagnostics in startup path
- `log_fiber_misalignment_stats(...)` prints angle stats wrt `[1,0,0]`.
- `validate_disarray_width_response(...)` performs synthetic dispersion sanity check to confirm larger width increases angle spread.

Important runtime robustness update:
- gather-based collective debug logging was removed from this function and replaced with rank-local logging to avoid MPI collective misuse in diagnostics.

---

## 4) MPI and startup/shutdown hardening (`MyoFE.py`)

### 4.1 FutureWarning suppression (FFC stack compatibility)
- Added:
  - `import warnings`
  - `warnings.filterwarnings("ignore", category=FutureWarning)`
- This suppresses noisy FFC/Numpy deprecation warnings without modifying installed FEniCS packages.

### 4.2 Fatal error handling for MPI runs
- Top-level execute path is wrapped in `try/except`.
- On fatal exception:
  - prints `Fatal error: ...`
  - calls `MPI.COMM_WORLD.Abort(1)`.

This prevents rank desynchronization and post-finalize crashes when one rank fails.

### 4.3 Removed explicit `MPI.Finalize()`
- Explicit finalization call was removed from application code to avoid finalize-order problems when libraries still use MPI.
- Shutdown is now managed by runtime/libraries.

### 4.4 `debug_mpi` flag
- Optional instruction flag `debug_mpi` added.
- When enabled, startup and shutdown diagnostics print:
  - rank
  - communicator type
  - `MPI.Is_finalized()` status.

---

## 5) Input configuration changes

### 5.1 New disarray-ready input file
- Added: `demos/fiber/Fiber_test_disarray.json`
- Based on prior fiber test input with disarray options enabled:
  - `mesh.fiber_architecture = ["disarray"]`
  - `mesh.disarray_width = [0.25]`
  - `mesh.disarray_seed = [42]`
- Legacy runtime reorientation sections were removed from this input for consistency with static-only behavior.

### 5.2 Updated output destinations
- Output paths in this disarray input were updated to:
  - data: `/mnt/gpfs2_4m/scratch/sba431/demos/fiber/sim_output/data_disarray.csv`
  - mesh: `/mnt/gpfs2_4m/scratch/sba431/demos/fiber/sim_output/mesh_output_disarray`

---

## 6) What was intentionally NOT changed
- No constitutive model equations were changed.
- No solver algorithm/newton logic changes were introduced.
- No passive model behavior was modified.
- No dynamic fiber remodeling logic remains active in the primary run path.

---

## 7) Practical usage guide

### 7.1 Aligned baseline
Use in mesh config:
```json
"fiber_architecture": ["aligned"]
```
Optional width/seed can be omitted.

### 7.2 Static disarray
Use:
```json
"fiber_architecture": ["disarray"],
"disarray_width": [0.25],
"disarray_seed": [42]
```

### 7.3 Baseline-equivalence check
Set:
```json
"fiber_architecture": ["disarray"],
"disarray_width": [0.0]
```
Expected behavior: preserves aligned `f0` baseline.

### 7.4 MPI debug
At root level instruction file:
```json
"debug_mpi": [true]
```

---

## 8) Troubleshooting notes

### 8.1 `Attempting to use an MPI routine after finalizing MPICH`
Mitigations now in place:
- no explicit `MPI.Finalize()` in app code,
- abort-on-fatal to keep ranks synchronized,
- no diagnostic gather collectives in disarray logging.

### 8.2 PETSc communicator errors (`.gather` missing)
Mitigation:
- removed gather requirement from disarray diagnostics; rank-local stats only.

### 8.3 FFC FutureWarning noise
Mitigation:
- startup `FutureWarning` suppression in `MyoFE.py`.

---

## 9) File-level change map
- `python_codes/LV_simulation/LV_simulation.py`
  - runtime fiber reorientation disabled and guarded.
- `python_codes/LV_simulation/mesh/mesh.py`
  - static fiber architecture initialization + one-time basis rebuild + diagnostics.
- `python_codes/MyoFE.py`
  - warning suppression, fatal abort handling, no explicit finalize, debug MPI flag.
- `demos/fiber/Fiber_test_disarray.json`
  - disarray-enabled runnable input and output path updates.

---

## 10) Validation summary (what has been checked)
- JSON syntax and key presence in disarray input.
- Presence of static-architecture init hook in `mesh.py`.
- Reorientation module disabled in `LV_simulation.py` runtime setup.
- Removal of explicit `MPI.Finalize()` and presence of abort-on-fatal in `MyoFE.py`.
- Absence of gather-based collectives in active disarray diagnostics.

This provides a static, deterministic-or-seeded disarray initialization pipeline suitable for MPI/PETSc runs without post-finalize MPI routine errors from diagnostics.
