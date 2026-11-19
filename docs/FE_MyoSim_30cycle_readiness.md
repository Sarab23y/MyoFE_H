# 30-cycle follow-up readiness review

**NOT READY for the full 30-cycle simulation. No simulation was submitted, committed or pushed.**

This extends the existing 15-file prototype in `work/MyoFE_H_30cycle_diagnostics`, detached at b5d051a14433c48257e1f6797863592dfd4d6889. The original diagnostic branch checkout is preserved. The isolated worktree has uncommitted changes; no branch was moved.

| Requirement | Classification | Evidence / remaining gate |
|---|---|---|
| Native Newton telemetry implementation | Implemented but not validated | Original native methods forwarded once; protocol tests pass. Actual image compilation, symbol preemption, callbacks, MPI collectives and numerical invariance unexecuted. |
| Source-stage event chronology | Implemented and validated on synthetic fixtures | Native samples, identities, stage ordering, missing data, jump intervals and +50 ms behavior tested; retained figures explicitly synthetic. Actual event chronology blocked by missing new stage recordings. |
| Original CG2 checkpoint validation | Implemented but not validated | Manufactured quadratic field on actual mesh; independent new-mesh reader and hash-bound reports prepared. DOLFIN execution unavailable. |
| Scalar/mixed/physical mapping at 1 and 32 ranks | Blocked by environment or missing data | Existing validators reused; exact handoff commands provided. No FE/MPI PASS claimed. |
| Physical invariance | Implemented but not validated | Three 200-step arms, exact initial-state comparison, per-step witness/enabled comparison and pure-control final/history comparison. No actual LV runs available. |
| Output-integrity checker | Implemented and validated on synthetic fixtures | Rejects missing reopen evidence, broken HDF/XML, bad timestamps and false interruption success. Actual FE files/flush behavior remain unvalidated. |
| 30-cycle termination | Implemented and validated with source-driver mock | Actual HR markers exercise original driver stop/final flush with mock FE. Actual 30-cycle run remains unexecuted and blocked. |
| Resource planning | Incomplete | Corrected partial storage envelopes; actual runtime, RSS, disk growth and final allocation not measured. |
| Existing three failures | Implemented and validated | Same failures in untouched checkout; no unrelated parameter or test edits. |

## Local validation

Final full suite: **231 tests; 228 pass, 3 fail, no errors**. Untouched checkout: **196 tests; 193 pass, the same 3 fail**. All 35 added tests pass. Logs are included. The two material-preservation failures arise from pre-existing `myofilaments/k_force` differences: generic base 0.01 versus 0.001 in five older validation inputs. The earlier report incorrectly called this k_1; this report corrects that identification. The third failure compares POSIX path strings with Windows backslashes. No related scientific inputs were changed to make tests pass.

Failing tests: `test_validation_inputs_preserve_baseline_material_parameters`, `test_cases_preserve_current_baseline_science`, and `test_range_case_generator_preserves_sources_and_sets_outputs`.

Synthetic examples are schema/algorithm fixtures, never historical populations or evidence of actual FE shortening. Local tests do not compile the C++ observer or validate DOLFIN semantics.

## Preserved calculations and evidence

`nsolver.py`, residual/Jacobian, Xi/ECM laws, MyoSim kinetics, quadrature degree, solver settings and production launchers remain unchanged. The two existing execution files contain tagged, opt-in observation hooks; tests verify that removing those blocks restores the base source exactly. The native observer is child-process-only LD_PRELOAD and delegates the original convergence/update methods; this design still requires proof against the actual container ABI. No replacement Newton algorithm is supplied. A minimal observed control is explicitly distinguished from a genuinely disabled arm.

See `changed_files/docs/FE_MyoSim_30cycle_followup.md` for implementation details and source citations, `source_manifest.json` for exact files/hashes, and the retained patches. Physical tolerances are atol=1e-10, rtol=1e-9; native residual/correction tolerances atol=1e-12, rtol=1e-9; IDs, initial state, iteration/status metadata are exact. Every discrepancy must be investigated, not waived.

## Resources and scientific limits

At the conservative 60,001-step cap, partial payload estimates with 50% margin are 177 GiB without event frames, 3.73 TiB under persistent triggering, plus an independently budgeted 560 GiB native JSON envelope assuming 50 Newton iterations at 32 ranks. These are not additive measured total guarantees; baseline/baroreflex logs, postprocessing and backend/Python memory are additional. The old 49-GiB no-event estimate omitted routine stage arrays and is superseded. Rolling array planning is 0.89 GiB globally, excluding object/backend RSS. No measured runtime estimate is available.

Actual historical events cannot be diagnosed from these synthetic fixtures. To establish physical FE deformation, reopen original mixed CG2 coefficients for event-adjacent accepted states with original mesh, reference fibers and function-space metadata, independently reconstruct F/J/stretch at validated physical coordinates, and compare against recorded values. Successful Newton convergence alone does not prove uniqueness/stability of equilibrium. No constitutive or physiological correction is proposed without that evidence.

Full-run readiness remains blocked until native telemetry, CG2 reopening, all mapping reports, paired physical invariance, output integrity and measured resources pass in the actual LCC environment. The full entry point remains intentionally blocked. No full submission command is provided.
