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

## 2. Existing displacement comparison (Windows/Python 3)

The postprocessor reads the existing `solution.xdmf` and associated HDF5 file
without opening either for writing.  Install dependencies in a VS Code Python
3 environment:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install numpy meshio h5py matplotlib
```

Run it against a copied or mounted simulation output directory:

```powershell
python validation\compare_cycle_displacements.py `
  --xdmf "D:\ECM_40s_healthy_quad_validation\sim_output\mesh_output\solution.xdmf" `
  --quadrature-coordinates "D:\ECM_40s_healthy_quad_validation\sim_output\quadrature_dof.npy" `
  --output "D:\ECM_analysis\displacement_cycles"
```

If `quadrature_dof.npy` is in a different output subdirectory, provide its
actual path.  Use `--displacement-field NAME` if the vector field is not named
`displacement` and automatic selection reports ambiguity.  The output
directory must not already exist.

The default cycle intervals are 37.484-38.316, 38.316-39.147, and
39.147-39.978 seconds.  Default phases are +0.10, +0.30, +0.50, and +0.70
seconds.  The script selects the closest saved snapshot independently for each
cycle, reports the exact requested/saved times, and requires the error to be
no greater than half the declared 0.1-second save interval.

Reports include raw displacement differences, residuals after a proper
best-fit rigid transformation, affine tetrahedral deformation-gradient and
determinant comparisons, and containing-element results for hotspot
Quadrature DOFs 1643, 1694, 2404, 4751, 4844, and 5004.  A Quadrature ID is
first converted through `quadrature_dof.npy` to a reference coordinate and
then located in a tetrahedron; it is never treated as a mesh-node ID.

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

The linear-tetrahedron `F` reconstruction is diagnostic.  For higher-order
CG2 displacement it uses the tetrahedron's four vertices and therefore cannot
recover within-element quadratic variation.  The 0.1-second save interval
also cannot resolve transitions occurring between snapshots.
