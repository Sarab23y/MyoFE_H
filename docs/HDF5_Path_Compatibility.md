# Job 6340723: diagnostic HDF5 filename compatibility

The failure names constructor argument 2, a C++ `std::string` filename. The
diagnostic used the three-argument call `HDF5File(comm, mesh_path, 'r')`, matching
the working production signature. It passed the original DOLFIN communicator.
The observed error points to filename conversion at the Python/C++ boundary;
it does not establish a communicator or HDF5 dataset defect.

## Trace and correction

Production `python_codes/MyoFE.py:execute_MyoFE` calls `json.load`, then
`dependencies/recode_dictionary.py:recode`. That recoder converts Python-2
Unicode values to UTF-8 `str` values. Production MeshClass then joins the mesh
path to its working directory and calls `HDF5File(mpi_comm_world(), path, 'r')`.

The diagnostic `resolve_mesh_path` reads `mesh.mesh_path[0]` with plain
`json.load`, resolves it relative to `REPO/python_codes` (or preserves an
absolute path), verifies the file exists and returns it. Under Python 2,
JSON text is Unicode and path operations preserve that type. A filesystem check
can succeed while a legacy SWIG `std::string` conversion rejects that value.
Under Python 3, JSON text is already native `str`.

`diagnostic_io.native_filename` now normalizes the representation at the
DOLFIN HDF5 boundary. Python 2 receives UTF-8 byte strings, matching production's
recoder. Python 3 receives native text. Byte input is decoded correctly on
Python 3, rather than converted into a literal string containing `b'...'`.
The resolved file, communicator object, argument order, read mode and dataset
names are preserved.

Both affected call sites use `open_hdf5_file`:

- `_production_mesh` in `validate_myosim_mixed_quadrature_ordering.py`, also
  reused by the geometry exporter.
- The geometry exporter's second HDF open for `ellipsoidal/endo_dist`.

The scalar quadrature validator builds a manufactured mesh and opens no HDF5
input. The search of mapping diagnostics found no other direct HDF constructors.
The preflight now performs a cheap real mesh/wall-data binding check before
the FE validator stages, including in `--static-only` mode. That mode still
performs no mechanical solve.

Immediately before construction, the log contains `[Diagnostic HDF5 constructor]`
with the Python version, original filename type, normalized filename type,
filename representation, communicator type, mode and constructor signature.
These target-container logs provide the actual types. The Python-2 Unicode
diagnosis is supported by the source difference and reported binding error;
this session cannot directly inspect the interpreter or arguments on LCC.

## Regression and runtime limits

Tests emulate a strict legacy binding that rejects the raw JSON string and
accepts its normalized UTF-8 representation, while verifying communicator
identity and argument order. The encoding is compared with the actual production
recoder method body. Other tests cover Python-3 text/bytes/path-like inputs,
non-ASCII paths, relative JSON paths through the complete shared loader, invalid
filenames, contextual constructor errors and the type log.

Local regression tests cannot certify the target DOLFIN constructor. The real
serial and 32-rank FE stages still require LCC execution. No production source,
scientific parameter, mechanical expression or production launcher is changed.

## Exact LCC rerun

Update the diagnostic worktree:

```bash
cd /home/sba431/MyoFE_H_diagnostics
git status --short
git pull --ff-only origin diagnostics/phase-regional-investigation
git rev-parse HEAD
export REPO=/home/sba431/MyoFE_H_diagnostics
export IMAGE=/home/sba431/fenics.img
```

Reuse the same exported `INPUT` and `DATA` paths that job 6340723 validated.
Those paths are recorded in its preflight stage commands. If this is a new
shell session, set them to those exact values before continuing:

```bash
: "${INPUT:?Set INPUT to the submitted instruction used by job 6340723}"
: "${DATA:?Set DATA to the saved-output directory used by job 6340723}"
export INPUT DATA
test -f "$INPUT"
test -d "$DATA"
export PREFLIGHT_ROOT=$(mktemp -d /mnt/gpfs2_4m/scratch/sba431/demos/ECM/hdf_preflight_6340723_XXXXXX)
```

First rerun the one-core preflight on the reported node/partition. The
constructor smoke check now precedes the numerical validator sequence:

```bash
PRE=$(sbatch --parsable --export=ALL --account=col_jfwe223_uksr \
  --partition=CAL48M192_D --nodelist=cascade001 --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G --time=00:30:00 \
  --output=hdf_preflight_%j.out --error=hdf_preflight_%j.err \
  --wrap='singularity exec --cleanenv "$IMAGE" python "$REPO/validation/preflight_diagnostics.py" --repo "$REPO" --input-json "$INPUT" --saved-output "$DATA" --output "$PREFLIGHT_ROOT/serial" --ranks 1')
PRE=${PRE%%;*}
echo "$PRE $PREFLIGHT_ROOT"
squeue -j "$PRE"
```

After completion, inspect the real types and the full serial verdict:

```bash
cat "$PREFLIGHT_ROOT/serial/mesh_binding.log"
cat "$PREFLIGHT_ROOT/serial/mixed_1.log"
cat "$PREFLIGHT_ROOT/serial/preflight.json"
sacct -j "$PRE" --format=JobID,State,ExitCode,Elapsed,NodeList
```

For Python 2 the normalized filename type should be `str`, with a UTF-8 byte
representation. For Python 3 it should also be `str`, with native text.
Require `passed: true` and `validated_ranks: [1]` before the 32-rank run.

```bash
cd "$REPO"
unset OUT
JOB=$(sbatch --parsable --export=ALL --dependency="afterok:$PRE" \
  --partition=CAL48M192_D --nodelist=cascade001 --nodes=1 --ntasks=32 \
  --cpus-per-task=1 --mem=32G --exclusive \
  validation/run_phase_mapping_validation.slurm)
JOB=${JOB%%;*}
export REPORTS=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/phase_mapping_${JOB}
echo "$JOB $REPORTS"
squeue -j "$JOB"
```

The command-line partition/node options override the diagnostic launcher's
SKY defaults. The new job ID defines a fresh report directory; existing
directories are rejected. No reports from jobs 6340723 or earlier are replaced.
If the preflight job has aged out of the scheduler, verify its successful
terminal state and report before submitting without that dependency.

After the mapping job finishes:

```bash
cat "$REPORTS/mixed_1.json"
cat "$REPORTS/mixed_32.json"
cat "$REPORTS/geometry_comparison.json"
cat "$REPORTS/mapping_manifest.json"
sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
```

The final manifest remains contingent on all ten reports passing. Constructor
success alone does not establish quadrature ordering or physical deformation.

The relevant SWIG Python-2 string conversion behavior is documented in the
[SWIG Python reference](https://swig.org/Doc4.1/Python.html).
