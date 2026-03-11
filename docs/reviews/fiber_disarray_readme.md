# Fiber Disarray in MyoFE (Static, Initialization-Only)

## Overview
This implementation perturbs the baseline fiber field once at initialization and keeps fibers frozen for the full simulation.

## Parameters (mesh block)
- `fiber_architecture`: `"aligned"` or `"disarray"`
- `theta_rms_deg`: RMS angular deviation from baseline fibers in degrees (default `0.0`)
- `ell_c`: correlation length for Gaussian perturbations (Helmholtz smoothing). Set `null` for uncorrelated noise (default `0.075`)
- `disarray_seed`: deterministic seed for reproducibility (default `1`)
- `disarray_region_mask`: `"auto"`, `"active_only"`, or `"all"` (default `"auto"`)

## Disarray model
Given baseline orthonormal triad `(fbar0, sbar0, nbar0)` and target `theta_rms` in radians:
- `w = theta_rms / sqrt(2)`
- sample `epsilon_s, epsilon_n ~ N(0, w)` (uncorrelated) or solve correlated fields via Helmholtz smoothing
- build `ftilde = fbar0 + epsilon_s*sbar0 + epsilon_n*nbar0`
- normalize `f0 = ftilde / ||ftilde||`

`f0` is then used by active stress as usual and is not time-updated.

## Correlated mode
If `ell_c` is set:
- solve `epsilon - ell_c^2 * Laplacian(epsilon) = xi` for each transverse component
- project to quadrature space
- globally rescale to target std `w` over the selected region

## Verification outputs
At initialization:
- norm safety checks and NaN guards
- misalignment stats wrt `[1,0,0]`
- RMS angle wrt baseline fibers vs target `theta_rms_deg`
- synthetic width-response check (`std@w` vs `std@2w`)

Runtime:
- frozen-fiber monitor checks that `f0` checksum remains unchanged over time.

## Quick sweep helper
Use `python_codes/run_fiber_disarray_sweep.py`:

```bash
python python_codes/run_fiber_disarray_sweep.py \
  --base-json demos/fiber/Fiber_test_disarray.json \
  --out-dir demos/fiber/sweep_cases \
  --thetas 15,20,25 \
  --ell-c 0.075,0.066,0.060
```

Add `--run --np 32` to execute immediately through MPI.
