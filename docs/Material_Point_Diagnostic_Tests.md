# Material-point diagnostic tests

These tests investigate two suspected causes of the healthy-cycle ECM spatial
instability.  They do not change or run the production model.

## 1. MyoSim mixed-Quadrature ordering (LCC/FEniCS 2017)

`validation/validate_myosim_mixed_quadrature_ordering.py` creates the same
degree-two scalar Quadrature element and repeated-scalar `MixedElement` used
for MyoSim populations.  It derives the component-to-parent mapping from each
cell and mixed subspace's actual DOLFIN dofmap.  Unique markers encode both the
global scalar Quadrature DOF and MyoSim state component.

The test performs both directions:

1. current production point-major packing -> authoritative mixed-subspace
   recovery; and
2. authoritative mixed-subspace assignment -> current point-major decoding.

It does not infer ordering from visually plausible arrays.

From the repository root on LCC:

```bash
mkdir -p demos/ECM/logs
sbatch validation/run_myosim_quadrature_ordering.slurm
```

Or run interactively inside the existing image:

```bash
INPUT="$PWD/demos/ECM/sim_input/base_instruction_Br_ECM_40s_healthy_quad_validation_optimized.json"
singularity exec --cleanenv /home/sba431/fenics.img \
  mpiexec -np 1 python validation/validate_myosim_mixed_quadrature_ordering.py \
  --input-json "$INPUT" --output myosim_order_1.json
singularity exec --cleanenv /home/sba431/fenics.img \
  mpiexec -np 32 python validation/validate_myosim_mixed_quadrature_ordering.py \
  --input-json "$INPUT" --output myosim_order_32.json
```

### PASS/FAIL

PASS requires, on every rank:

- mixed owned DOFs = scalar owned DOFs times the population-component count;
- the production `4 * mesh.num_cells()` point count equals the scalar
  Quadrature vector's owned length on every rank;
- no conflicting or duplicate subspace-to-parent mappings;
- no missing owned parent DOFs;
- zero forward mismatches; and
- zero inverse mismatches.

A FAIL proves that the current point-major packing does not represent the
actual mixed-space layout in that runtime/partition.  A PASS establishes the
tested vector-layout correspondence, but does not establish correct MyoSim
kinetics, FE mechanics, or cycle stability.

## 2. Existing displacement comparison (LCC/FEniCS image)

The Python 2.7 postprocessor reads the existing `solution.xdmf` XML and only
the HDF5 datasets referenced by XDMF Attributes named `displacement`. Other
fields at the same timestamp are counted and skipped, not treated as duplicate
snapshots. Input files are never opened for writing.

From the repository root:

```bash
git pull --ff-only
mkdir -p demos/ECM/logs
sbatch validation/run_cycle_displacement_analysis.slurm
```

The launcher defaults to the completed run at
`/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_healthy_quad_validation_optimized`
and writes a new job-specific directory named
`displacement_cycle_analysis_${SLURM_JOB_ID}`. Override `BASE` or `OUT` when
submitting if necessary; an existing output path is refused.

After SLURM reports completion, inspect the machine-readable results with:

```bash
OUT=/mnt/gpfs2_4m/scratch/sba431/demos/ECM/displacement_cycle_analysis_JOB_ID
python -m json.tool "$OUT/displacement_cycle_comparison.json" | less
column -s, -t "$OUT/selected_snapshots.csv" | less -S
find "$OUT" -maxdepth 1 -type f -name '*.png' -print
```

The default cycle intervals are 37.484-38.316, 38.316-39.147, and
39.147-39.978 seconds.  Default phases are +0.10, +0.30, +0.50, and +0.70
seconds.  The script selects the closest saved snapshot independently for each
cycle, reports the exact requested/saved times, and requires the error to be
no greater than half the declared 0.1-second save interval.

Reports include raw displacement differences, residuals after a proper
best-fit rigid transformation, deformation-gradient and determinant
comparisons, and containing-element results for hotspot
Quadrature DOFs 1643, 1694, 2404, 4751, 4844, and 5004.  A Quadrature ID is
first converted through `quadrature_dof.npy` to a reference coordinate and
then located in a tetrahedron; it is never treated as a mesh-node ID.

For an XDMF `tetra10` geometry with displacement values at all ten nodes, the
script evaluates the quadratic shape-function gradient at the hotspot's
barycentric coordinate. For a four-node visualization mesh it reports the
explicit fallback `linear_tetra4`. A normal `XDMFFile.write()` visualization
does not carry a reconstructible DOLFIN CG2 Function/dofmap checkpoint; the
script therefore does not claim an exact CG2 gradient when only vertex data
were saved.

### PASS/FAIL

PASS means that the XDMF data are finite, timestamps are unique, and every
requested phase has a valid in-cycle snapshot within the expected sampling
error.  Mechanical differences are measurements rather than pass thresholds:

- a large raw difference that disappears after rigid alignment supports
  rigid drift;
- a persistent rigid-aligned difference and differing element `F` support a
  genuine FE deformation difference;
- matching displacement/`F` with differing saved Quadrature stretch would
  support a downstream projection or mapping problem.

The 0.1-second save interval cannot resolve transitions occurring between
snapshots. The JSON records exact requested times, selected snapshot times,
and signed phase errors so this limitation remains visible.
