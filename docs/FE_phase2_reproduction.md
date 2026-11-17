# Phase 2 evidence and reproduction

Start with `Phase2_Report.md`. No historical FE deformation has been recovered.
Source-law numerical tests are independent local assemblies, not DOLFIN/MPI or
the full LV problem. Archived scalar inputs and synthetic states are labeled.

## Contents

- `element_probe/`: volume energies/residuals/tangents/JVP convergence, UFL/FD
  checks, numeric source-law evaluator and synthetic nodal states.
- `cavity_probe/`: source-equivalent CG2/Real face integration convergence.
- `compiler_audit.json` and `.log`: FFC effective integral metadata under the
  inspected compiler overrides, not the historical image's compiled kernels.
- `active_shear_verified/`: conditional force moments, source-UFL acoustic
  checks, reduced shear roots, Newton traces and a residual curve. Its initial
  two-bin vector is only a moment representative, never historical populations.
- `verification_final/tests.json`: final 69 focused checks, all passed.
- `existing_tests.txt`: 92 existing tests, all passed.
- `conditional_*.csv`: explicitly conditional inferences from archived scalars.
- `candidate_*` and `input_reaudit/`: coordinate-set and input-basis checks,
  not production-rank/component identity validation.
- `scripts/`: six uncommitted diagnostic scripts. No production code changes.
- `archived_event_inputs.csv`: copy of the previous phase's native event
  extraction, retaining its stage/provisional-mapping limitations.
- `saved_reference_coordinates.npy`: copied reference point cloud for the
  geometric-set test. This is not a verified global-ID or fiber-component map.
- `candidate_configuration.json`: candidate parameters only, not an
  authenticated submitted instruction.
- `LCC_Readonly_Commands.md`: unchanged guide from the previous phase. Check
  existing job status and original-file integrity before considering a rerun.

`verification/` and `verification_complete/` retain earlier successful checks;
the final suite is `verification_final/`. An empty `active_shear/` directory, if
present locally, is an earlier diagnostic serialization failure; the successful
result is `active_shear_verified/`. No numerical result from the failed write
is presented as a passing test. The final ZIP need not include empty directories.

## Environment actually used

Python 3.7.16, NumPy 1.16.5, SymPy 1.4, UFL/FFC/FIAT 2019.1.0, with SciPy,
h5py and Matplotlib. Exact versions and hashes are in `provenance.json`.
No DOLFIN runtime was installed or used. Do not assume these scripts run
unchanged with modern replacements for the 2019 FEniCS Python libraries.

## Reproduce the local numerical experiments

Use a separate existing environment and a source copy pinned to
`b5d051a14433c48257e1f6797863592dfd4d6889`. Do not modify a production checkout
or install packages into its runtime. No network fetch, installation or
simulation submission is included here. Supply the existing candidate mesh
file recorded by hash in `provenance.json`; LFS pointer text is not a mesh.

Example PowerShell commands, adapting only the bundle/source locations:

```powershell
$py = 'C:\ProgramData\Anaconda3\python.exe'
$bundle = 'C:\Users\sba431\Documents\Codex\2026-11-10\files-pasted-by-the-user-myofe\outputs\fe_phase2_numerical_investigation'
$repo = 'C:\Users\sba431\Documents\Codex\2026-11-10\files-pasted-by-the-user-myofe\work\MyoFE_H'
$mesh = Join-Path $repo 'demos\HCM_paper\simulations_baseline_final\baseline_final_Kappa_4_k1_3.7\sim_inputs\ellipsoidal.hdf5'
$scripts = Join-Path $bundle 'scripts'
$out = Join-Path $bundle ('reproduction_' + [guid]::NewGuid().ToString('N'))

& $py "$scripts/phase2_element_probe.py" --repo $repo --mesh $mesh --output "$out/element_probe"
& $py "$scripts/phase2_compiler_audit.py" --repo $repo --output "$out/compiler_audit.json"
& $py "$scripts/phase2_cavity_probe.py" --mesh $mesh --output "$out/cavity_probe"
& $py "$scripts/phase2_active_shear_probe.py" --repo $repo --event-csv "$bundle/archived_event_inputs.csv" --output "$out/active_shear"
& $py "$scripts/phase2_input_checks.py" --mesh $mesh --coordinates "$bundle/saved_reference_coordinates.npy" --output "$out/input_audit"
& $py "$scripts/phase2_verify_results.py" --probe "$out/element_probe" --shear "$out/active_shear" --input-audit "$out/input_audit" --output "$out/verification"
```

Inspect each exit code before continuing. Outputs use fresh directories and
must not be placed under original simulation results. This local sequence is
short (roughly minutes), not a complete ventricular simulation. Compiler log
verbosity is expected. The core conditional determinant/rank-one historical
inferences use the equations and assumptions printed in the report; they are
not exact F reconstruction or full-state replay.

The `element_probe/local_reference_arrays.npz` contains the last (distorted)
reference fixture's arrays. All state vectors/parameters are in its JSON and
can be reassembled with the supplied code; it is not a full multi-state archive
of every matrix or a historical checkpoint.

## Data still required

The original submitted instruction/source/runtime, full-order CG2/CG1 states,
reference basis and material fields in authoritative DOF order, pre-reset
cellular/moment/reference state and original Newton logs. The current reference
point-cloud match and candidate-file basis test do not establish those facts.
