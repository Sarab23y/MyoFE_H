# Review: `python_codes/LV_simulation/dependencies` (`.py` files)

## Scope
This review covers all Python files under `python_codes/LV_simulation/dependencies`:
- `forms.py`, `forms copy.py`
- `nsolver.py`, `nsolver_old.py`, `nsolver_original.py`
- `solver.py`
- `assign_heterogeneous_params.py`
- `assign_local_coordinate_system.py`
- `recode_dictionary.py`, `recode_json_strings.py`
- `batch_generator.py`
- `__init__.py`

## Structure and logic summary

### 1) `Forms` class (`forms.py` / `forms copy.py`)
**Role**
- Encapsulates continuum mechanics helper expressions and assembled quantities used by the LV solve: deformation tensors (`Fmat`, `Fe`, `Emat`, `Cmat`, `J`), cavity volume/pressure helpers, passive constitutive terms, and active stress scaffolding.

**Strengths**
- Good concentration of mechanics formulas in a dedicated class, reducing equation sprawl in the simulation driver.
- Parameterized design (`self.parameters`) lets the same class work across LV/RV and optional growth formulations.

**Improvement suggestions**
- Remove duplicate file maintenance (`forms.py` vs `forms copy.py`) and keep one canonical implementation.
- Add explicit parameter validation in `__init__` (required keys, expected tensor/function space types), to fail early with descriptive errors.
- Normalize style inconsistencies (mixed tabs/spaces, mixed legacy Python-2/3 style comments/prints in related modules).
- Split very large constitutive functions (e.g., passive law branches) into smaller private methods per law to improve testability and readability.
- Minimize repeated `assemble(...)` in hot paths when values can be cached per Newton iteration.

### 2) `NSolver` class (`nsolver.py`, plus historical variants)
**Role**
- Central nonlinear solve manager for the coupled mechanics/growth system.
- Supports standard `solve(...)` and a debugging-heavy custom residual/Jacobian loop.

**Strengths**
- The debug path includes valuable NaN diagnostics and per-term residual reporting that is useful for difficult constitutive failures.
- Solver parameters are configurable from instruction input.

**Improvement suggestions**
- Consolidate `nsolver.py`, `nsolver_old.py`, and `nsolver_original.py` to one maintained implementation; keep old versions in version control history, not runtime tree.
- Move debug diagnostics into dedicated helper functions to shorten `solve_growth` and reduce branch complexity.
- Replace print-based diagnostics with a small rank-aware logger utility.
- Avoid hardcoded `debugging_mode = True`; use parsed solver parameters consistently.
- Add guardrails around optional expressions referenced only in debug (there are references to `f2_temp`/`f3_temp` paths that are fragile in partial-assembly contexts).

### 3) `Problem` / `CustomSolver` classes (`solver.py`)
**Role**
- Thin wrappers around FEniCS `NonlinearProblem` and `NewtonSolver` setup.

**Strengths**
- Clear separation of residual/Jacobian assembly from high-level simulation class.

**Improvement suggestions**
- `CustomSolver.__init__` references `mesh.mpi_comm()` without a local `mesh` symbol. Inject a communicator (or mesh) explicitly through constructor arguments.
- Move PETSc options into a configurable parameter object rather than hardcoding `gmres`/`ilu` globally.
- Add class/module docstrings describing expected lifecycle and ownership of matrices/vectors.

### 4) `assign_heterogeneous_params` class (`assign_heterogeneous_params.py`)
**Role**
- Parses heterogeneous-law directives from nested configuration and applies spatial heterogeneity to dolfin functions (and historically to HS parameter lists).

**Strengths**
- Supports a broad set of heterogeneity laws (`gaussian`, `percent_fibrosis`, `transmural`, infarct-related options), which is valuable for physiological modeling flexibility.

**Improvement suggestions**
- Refactor monolithic law dispatch into a dictionary of handlers (`law_name -> function`) to simplify extension and reduce branching.
- Validate law-specific required fields up front and raise clear exceptions.
- Resolve scope/undefined-variable fragility (e.g., `no_of_int_points`/`geo_options` usage in some branches not passed in the local function signature).
- Rename class to `AssignHeterogeneousParams` (PEP8) and standardize method naming.
- Add deterministic random-seed plumbing for reproducibility in stochastic laws.

### 5) Coordinate assignment utilities (`assign_local_coordinate_system.py`)
**Role**
- Builds local fiber/sheet/sheet-normal directions for multiple geometries and updates geometry markers used by downstream heterogeneity assignment.

**Strengths**
- Handles multiple geometry modes and can load precomputed fibers when available.

**Improvement suggestions**
- Split the giant geometry `if` chain into per-geometry handler functions (`_assign_for_ventricle`, `_assign_for_cylinder`, etc.).
- Reduce repeated projection/interpolation patterns by creating small reusable helpers.
- Separate side effects (`geo_options` mutation) from pure coordinate calculations for easier testing.
- Improve variable naming and remove dead/commented legacy code blocks that obscure active logic.

### 6) JSON/string recoding utilities (`recode_dictionary.py`, `recode_json_strings.py`)
**Role**
- Legacy Python-2 unicode-to-bytes conversion utilities for JSON-loaded data.

**Improvement suggestions**
- In Python 3 these are largely obsolete and can be removed or replaced with a no-op compatibility layer.
- If retained for compatibility, add runtime/version guards and tests proving behavior for modern `str` inputs.
- Avoid duplicate utilities with nearly identical behavior; keep one module.

### 7) Batch job script generator (`batch_generator.py`)
**Role**
- Script-style utility to generate instruction JSONs and SLURM job scripts for core-scaling runs.

**Improvement suggestions**
- Wrap script logic in `main()` and add CLI arguments (template path, output root, core list).
- Avoid in-place mutation aliasing (`temp_instruction = json_file`) by deep-copying per iteration.
- Fix path concatenation bugs from reusing `new_json_str` as both filename and full path.
- Extract cluster-specific constants (account, partition, singularity image) into config inputs.

## Cross-cutting recommendations
1. **Canonicalize active modules**: keep one `forms` and one `nsolver` implementation.
2. **Adopt lightweight typing/schema checks** for nested instruction dictionaries.
3. **Introduce rank-aware logging** and remove scattered print statements.
4. **Add focused regression tests** for heterogeneity law assignment and key solver setup invariants.
5. **Modernize Python compatibility** by removing Python-2-only unicode helpers and style remnants.

## Suggested refactor order (low risk)
1. Remove/retire duplicate legacy files (`* copy.py`, `*_old.py`, `*_original.py`) after confirming import points.
2. Add validation and logging helpers without changing solver math behavior.
3. Refactor large dispatcher methods into handler maps.
4. Add tests for heterogeneity and coordinate assignment determinism.
5. Parameterize cluster/script generation and clean path handling.
