# Review: `python_codes/LV_simulation/fiber_reorientation/fiber_reorientation.py`

## Scope
This review focuses on the `fiber_reorientation` class in `fiber_reorientation.py`, summarizing structure/logic and recommending concrete improvements.

## Structure and logic summary

### Class role
`fiber_reorientation` is a remodeling helper attached to the LV simulation object. Its core responsibilities are:
1. Read reorientation configuration from `instruction_data`.
2. Build a driving signal from stress fields.
3. Compute a per-step fiber adjustment vector (`f_adjusted`) via a stress law.
4. Rebuild local material coordinate axes (`f0`, `s0`, `n0`) after fiber updates.

### Method-level behavior

#### `__init__(self, parent_params)`
- Stores a pointer to the parent LV simulation object.
- Copies `fiber_reorientation` config entries into `self.data`.
- Resolves a stress-based driving signal via `return_driving_signal(...)`.
- Reads `time_step` from protocol and computes initial `f_adjusted` by calling `stress_law(...)`.

#### `stress_law(self, s, time_step, function_space)`
- Uses parent FE fields to compute reorientation direction.
- Current active path forms orientation tendency from **active + passive** stress directions:
  - `f_active = normalize(Pactive * f0)`
  - `f_passive = normalize(passive_total_stress * f0)`
  - `f = f_active + f_passive`
- Projects `f` to DG vector space, then computes incremental adjustment:
  - `f_adjusted = (1/kappa) * (f_proj - f0) * time_step`
- Projects adjustment to provided function space and returns it.
- Contains additional inactive/commented variants (FR coefficients, traction weighting, debug traces).

#### `return_driving_signal(self, signal_type)`
- Maps `signal_type` to a mesh stress field (`total_stress` or `total_passive_PK2`).
- Returns UFL/FEniCS field handle.

#### `update_local_coordinate_system(self, fiber_direction)`
- Pulls vector data arrays for `f0`, `s0`, `n0`.
- Re-normalizes each local fiber direction triplet.
- Reconstructs `s0` via cross product with global z-axis.
- Reconstructs `n0` via cross product of `f0` and `s0`.
- Returns updated arrays (`s0`, `n0`, `f0`) for assignment upstream.

## Strengths
- Good conceptual separation: signal selection, reorientation law, and local-basis update are explicit.
- Uses FE projections to move between symbolic expressions and concrete field data.
- Includes practical hooks for growth/remodeling coupling (active/passive split).

## Concrete improvement suggestions

### 1) Remove ambiguity between configured and actual driving signal
- `stress_law` accepts `s` but active implementation mostly ignores it (uses active/passive fields directly).
- Either:
  - make `s` the true source of direction, or
  - rename API to reflect hardcoded active/passive blending.

### 2) Add robust normalization guards
- Multiple operations divide by vector norms (`sqrt(inner(...))` and `np.inner(...)`) without epsilon checks.
- Add `eps` safeguards to avoid NaNs when stress vectors or local basis vectors are near zero.

### 3) Clean dead/commented code blocks
- Large commented debug/alternative branches make behavior hard to verify.
- Move alternatives into named strategy methods (e.g., `law_basic`, `law_weighted`, `law_capped`) selected by config.

### 4) Validate configuration keys and values
- Fail fast if required keys are missing (`time_constant`, `stress_type`), or invalid (`time_constant <= 0`).
- Return explicit errors for unsupported `stress_type` values.

### 5) Improve naming consistency and readability
- Rename class `fiber_reorientation` -> `FiberReorientation` (PEP8) if refactor scope allows.
- Fix typo variable names (`f_actvie` -> `f_active`).
- Add short docstrings for each method with input/output expectations.

### 6) Make debug logging rank-aware and configurable
- Replace ad-hoc `print(time_step)` / `print("test3")` with a logger gate based on a verbosity/debug flag.
- Prefix logs with module name and rank for MPI traceability.

### 7) Separate numerical kernel from FE plumbing
- Extract pure-vector basis update and normalization into utility functions.
- Keep FE `project(...)` calls in thin wrappers; this improves testability and allows unit tests without full FEniCS context.

### 8) Revisit coordinate-system update assumptions
- `update_local_coordinate_system` assumes global z-axis for sheet construction, which can degenerate when `f0` aligns with z.
- Add fallback reference axis selection to avoid near-singular cross products.

### 9) Clarify FR constraints and caps
- The file defines `FR_max`, `FR_coeff`, and weighting ideas, but active path currently ignores them.
- Implement one clearly documented stabilization approach (angular cap or coefficient) and expose it via config.

### 10) Add targeted tests
Suggested minimal tests:
- zero-stress case should produce finite, near-zero update;
- near-collinear basis update should remain finite;
- unsupported `stress_type` should raise clean error;
- `kappa` and `time_step` scaling should be monotonic and predictable.

## Suggested low-risk refactor order
1. Add config validation and normalization guards.
2. Remove/relocate dead commented branches.
3. Extract basis-update helpers and add unit tests.
4. Introduce configurable stress-law strategy selection.
5. Standardize logging and naming cleanup.
