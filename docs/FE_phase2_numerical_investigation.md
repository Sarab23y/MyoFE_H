# MyoFE_H Phase 2 — FE numerical artifact investigation

Follow-up to the 17 November 2026 FE-first investigation. Diagnostic only.

## Main conclusion

**The source mechanics can exhibit a concrete local instability mechanism in a controlled test. The historical initiating cause is still unproved.** This phase goes beyond the earlier polynomial-order example:

1. Source-law CG2/CG1 local integration tests show degree-2 residual errors up to **0.13%** and tangent errors up to **1.50%** in the tested smooth synthetic states. A source-equivalent cavity-face test has tangent errors up to **0.96%**. Higher-degree rules converge, and directional-derivative checks agree. These are observed synthetic integration errors, not a demonstrated incorrect historical Newton solve.
2. With the 2833 event's *conditionally reconstructed aggregate force moments* and an explicitly synthetic J=1 geometry, the source UFL tangent has an **isochoric rank-one shear coefficient of -94.05 Pa**. Passive ground matrix contributes +125.78 Pa and the pre-reset active scalar contributes -219.84 Pa. Xi is inactive. This is a demonstrated source-model loss of local rank-one stiffness for that fixture, despite positive J and nonnegative moment-representative populations.
3. The corresponding reduced shear equilibrium has roots at gamma=0 and approximately **±0.129808**. Newton accepts the zero-residual but negative-curvature zero root, or reaches either sheared root from different initial guesses. This is a **one-mode constrained experiment**, not an assembled full-LV branch-switching demonstration.
4. There are two strong conditional historical checks. Under the reviewed parameter/stage/mapping/basis assumptions, the archived scalar data give a negative rank-one stiffness certificate at 2833 during **8.688–8.697 s**, before negative overlap. At minimum stretch, the archived ECM scalar and the candidate ground law imply **|J|≤0.24131**. These checks could alternatively expose a failed premise: different parameters, non-unit/non-orthogonal fibers, inconsistent field recording or incorrect point identity. Neither is an independent measurement of original F or J.

The smallest decisive next experiment is therefore **not a full simulation**. Authenticate the original input, reference basis and point identities, then obtain one full-order mechanical state near 8.696–8.707 s and evaluate J, the source-law rank-one tangent and the pre-reset mechanical residual. This will test the two conditional checks directly.

## Continuity, evidence and scope

The prior report, evidence ZIP and LCC read-only guide were present and read. The ZIP contains 29 members and passes its integrity check. Its conclusions about stage timing, negative-overlap chronology, damaged Windows visualization files and missing historical-state provenance are retained, not rediscovered or silently replaced.

The original diagnostic checkout and the read-only GitHub branch both remain at `b5d051a14433c48257e1f6797863592dfd4d6889`. An isolated detached worktree, `work/MyoFE_H_fe_phase2`, contains six new **uncommitted diagnostic scripts**. No tracked production file was changed. No commit, push, merge, PR or LCC submission occurred. The first worktree checkout encountered an LFS credential/download failure; it was recreated without LFS downloading. The actual candidate HDF5 mesh was read from the previously available original local diagnostic checkout, not from an unresolved LFS pointer.

Local DOLFIN is unavailable. **UFL, FFC and FIAT 2019.1.0 are available**, alongside NumPy/SciPy/SymPy. Tests below execute actual source-law function bodies, actual UFL differentiation, actual FFC analysis and FIAT basis/quadrature tabulation. The local matrix assembly is an independent Python assembler. It is not DOLFIN assembly, an MPI run, an original-state reconstruction or the complete LV system.

Evidence labels:

- **Source/FFC-confirmed:** behavior of inspected source and compiler analysis with stated parameters.
- **Executed synthetic/local:** controlled state, coefficient extension or reduced-mode experiment.
- **Archived scalar:** supplied saved CSV values, retaining their original timestamps and provenance limitations.
- **Conditional inference:** depends on explicitly listed producer, parameter, basis and mapping premises.
- **Blocked:** requires original LCC/runtime/full-order data.

The actual submitted instruction, original producer snapshot, original FEniCS runtime parameters and complete event checkpoints remain unavailable. The repository candidate is not authenticated as the historical submitted input. No original LCC files or scheduler status could be inspected here.

## 1. FE quadrature audit

Source locations are pinned to the inspected commit. Paths in this section are relative to `python_codes/LV_simulation/`.

| Term | Exact source | Degree evidence |
|---|---|---|
| Global compiler defaults | `mesh/mesh.py`, `initialize_function_spaces`, 90–92 | Sets `quadrature_degree=2` and representation `quadrature`. |
| Displacement/pressure spaces | Same function, 98–127 | CG2 vector displacement, CG1 pressure, Real multipliers, degree-2 scalar/mixed Quadrature state. |
| Configured spaces | Same function, 143–177 | Instructions can replace named spaces. The candidate's material-vector and scalar Quadrature spaces are both degree 2; DG output spaces do not set mechanics integration. |
| Xi, ground, collagen energy | `dependencies/forms.py`, `_passive_energy_components`, 213–258; weighting 260–266 | Nonpolynomial laws; assembled through `mesh.py:1005–1009` on bare `dx`, without a higher-degree measure override. |
| Pressure constraint | `forms.py`, `PassiveMatSEF`, 268–273 | `-p*(J-1)` is included in the same energy derivative/F1, using the same `dx`. |
| Active virtual work | `mesh.py:1086–1097,1133–1136`; `return_cb_stress:1303–1334` | `inner(F*Pactive,grad(v))*dx`, no separate integration metadata. |
| Cavity constraint/external cavity pressure | `forms.py`, `LVV0constrainedE`, 113–140; `mesh.py:1137–1142` | Surface cofactor/position integral on `ds(endo)`. Area uses `assemble(..., representation=uflacs)` without overriding degree. The pressure-loaded growth variant is separate from the baseline volume constraint. |
| Rigid constraints | `mesh.py:1145–1150` | Real multipliers with displacement and reference-coordinate moments on `dx`; global constraints, not a local hourglass stabilization. |
| Optional spring | `mesh.py:1170–1183` | Boundary spring term, no higher-degree override. Absent in the candidate input; historical activation requires its actual instruction. |
| Jacobian | `mesh.py:1155–1167,1180–1183` | UFL derivatives of the same residual forms. Differentiation does not itself increase the selected rule. |
| Ordinary nonlinear assembly call | `dependencies/nsolver.py:337–346` | Overrides representation to `uflacs`, but supplies no quadrature degree override. |

This is more than an inference from `deg=2`. `phase2_compiler_audit.py` constructed the source-law UFL forms with the matching CG2/CG1/Real/Quadrature coefficient types and analyzed each separately using FFC 2019.1.0 and the reviewed degree-2 override. **All cell and exterior-facet integrals analyzed as default-rule degree 2**, including the total Jacobian and optional spring. `compiler_audit.json` retains the integral metadata and `compiler_audit.log` the compiler output.

This establishes the compiler result **for those settings**. It is not a capture of the historical image's compiled kernels or global parameter tree. That last step still requires LCC provenance.

### Expected order limitations

- On an affine tetrahedron, CG2 u gives linear grad(u) and generally cubic J. `q*(J-1)` with CG1 q is generally quartic. Pressure-related displacement tangents can also exceed degree 2, including a linearly varying pressure at an undeformed state.
- Xi, collagen, ground exponentials and stretch-dependent active work are not generally integrated exactly by a finite low-degree polynomial rule.
- The cofactor cavity integrand contains quadratic cofac(F) times quadratic `(X+u)`, generally degree four on a face. Rigid rotation moments can contain cubic polynomials. Optional spring terms may be higher order still.
- These limitations do not automatically demonstrate an unstable discrete model, a wrong algebraic Jacobian or a historical event cause.

## 2. Executed integration-convergence tests

### Actual geometry, synthetic fields

The candidate HDF5 mesh has 442 vertices and 1252 tetrahedra. Its four degree-2 quadrature locations per cell match the supplied 5008-coordinate point cloud bijectively to **2.23e-16**. This verifies geometric-set correspondence only; it does not verify MPI global IDs, component packing, ownership, fiber basis or the historical producer.

A separate candidate-file basis check used the stored per-cell DOF lists and
the matching FFC/FIAT vector-Quadrature component-block layout. Across 5008
points, maximum norm error is 8.2e-8, pairwise dot magnitude 7.3e-8, and basis
determinants are within about 1.5e-7 of one. This supports a near-orthonormal
**input** basis under that layout; it does not certify production MPI transfer
or the historical output mapping. It is not a naive reshape of the global
vector, and does not replace the existing vector-mapping validator.

The volume comparison uses one median-volume tetrahedron from this candidate mesh. Displacement is represented in the **actual FIAT CG2 basis** and pressure in CG1. The same local node-major ordering and no local boundary constraints are used at every rule degree. The operators contain passive, active and mixed pressure/constraint terms. The reported local “total” is that **volume restriction**, not the full global LV residual with cavity and rigid multipliers. A separate candidate endocardial face exercises the cavity term.

All reference fibers are explicitly synthetic unit coordinate axes. Material coefficients are constant candidate values, hsl0=1000 nm, bound fraction=.005 and smooth synthetic HSL-reference/pressure fields. Those choices define values at higher quadrature points. They are not an interpolation or reconstruction of historical Quadrature data. Simply increasing quadrature degree in production would not define its cellular/material coefficients at new integration points and can fail during Quadrature-element compilation.

Five synthetic states were tested:

| State | Central F and spatial variation |
|---|---|
| Reference | F=I; zero displacement. Pressure varies linearly in this fixture, so “reference” is not stress-free equilibrium. |
| Diastolic proxy | Diagonal (1.08,1.03,1/(1.08×1.03)), small quadratic displacement variation. |
| Systolic proxy | Diagonal (.82,1.08,1/(.82×1.08)), small quadratic variation. |
| Compressed fiber | Diagonal (.55,1.24,1/(.55×1.24)), small quadratic variation. |
| Severely distorted | Upper triangular with diagonal (.35,1.65,1/(.35×1.65)) and shears .35,.10,.25, plus small quadratic variation. |

The proxies use plausible directional magnitudes for controlled testing, not validated patient/run states. All sampled determinants remain positive; minimum J over degree-10 points is about 0.978–0.981 in the nonreference fixtures. Different rules sample different locations, so their sampled minimum J values are not conflicting reconstructions.

FIAT rules contain 4, 14, 24 and 125 tetrahedral points at degrees 2,4,6,8. Degree 10 is the comparison reference; degree 12 independently checks convergence. Maximum degree-10/12 residual difference is below 5e-15 and tangent difference below 3.2e-14 in these fixtures.

### Scale-aware definitions

Native mixed residual and tangent entries have different units. With L=`|det(B)|^(1/3)` from the reference cell and stress scale P*=1000 Pa, displacements are scaled by L, pressures by P*, displacement residuals by P*L² and pressure residuals by L³. Matrix scaling uses the same row scales and unknown scales. Reported errors are `||scaled difference||/||scaled high-degree reference||`; vectors use Euclidean norms and matrices Frobenius norms. Individual energy/work absolute and relative errors and raw residual differences are also supplied. Near-zero energy has no meaningful relative error and is left undefined.

Absolute energies have units Pa times the mesh-coordinate volume, and weak-form residuals retain their native coordinate units. They should not be called joules/newtons without authenticating the model's coordinate-unit convention.

### Degree-2 volume results versus degree 10

| Synthetic state | Scaled residual relative error | Scaled tangent relative error | Selected JVP relative error |
|---|---:|---:|---:|
| Reference | 5.1e-15 | 4.43e-5 | 4.96e-5 |
| Diastolic proxy | 7.46e-5 | 9.67e-4 | 5.24e-4 |
| Systolic proxy | 7.37e-4 | 1.89e-3 | 2.40e-3 |
| Compressed | 9.85e-4 | 1.35e-2 | 1.34e-2 |
| Distorted | 1.31e-3 | 1.50e-2 | 1.49e-2 |

The undeformed tangent difference is not automatically a threshold bug: pressure varies in that fixture. An independent constant-pressure reference control makes degree-2 and degree-10 tangents agree within 1e-12.

For the distorted state:

| Degree | Scaled residual error | Scaled tangent error |
|---|---:|---:|
| 2 | 1.31e-3 | 1.50e-2 |
| 4 | 5.21e-6 | 8.86e-5 |
| 6 | 2.22e-8 | 4.51e-7 |
| 8 | 1.05e-12 | 3.39e-11 |

Full five-state tables, every degree, each constituent energy, active virtual work, individual residual blocks, raw absolute differences and JVP errors are in `element_probe/integration.csv` and `energy_convergence.csv`. Passive energy error at degree 2 is smaller than tangent error here: about 1.34e-5, 2.78e-5, 2.19e-5 and 8.88e-7 relative for the four deformed states. Small energy error does not certify the tangent.

The candidate endocardial-face test uses the source cofactor cavity integrand and CG2/Real variations. Degree-2 tangent errors range from **0.58% to 0.96%** for its deformed fixtures (0.62% at reference). Degree 4 agrees with degree 10 within about 3e-15. This is local surface integration, not global cavity-pressure equilibrium; prescribed-volume/area terms were constant in this fixture.

### Jacobian consistency, not just integration order

Actual source-law function bodies were differentiated using UFL. The independent numerical constitutive derivatives agree with UFL Piola values within 3.3e-15 and UFL JVPs within 1.5e-16 relative in the four nonthreshold states. UFL JVP versus central finite difference differs by at most 2.7e-8 in the tested directions.

For the independently assembled nodal mixed operator, decreasing the scaled directional-difference step from 1e-4 to 1e-6 reduces relative error from at most 1.8e-6 to at most 1.7e-10. The constitutive branch is held away from Xi/collagen thresholds for these tests. Thus the tested tangent is consistent with its fixed-rule residual; this is **not evidence of an AD Jacobian coding error**. It also does not authenticate the historical global Jacobian, MPI ordering or boundary assembly.

### Required integration conclusion

1. Mathematical order limitation: confirmed.
2. Observed non-negligible integration differences in synthetic local tests of the actual source algebra: confirmed.
3. Incorrect original global production FE formulation/integration: **not established**. Spatial Quadrature coefficients, actual fields, boundary terms and runtime kernels must be checked on authenticated data.
4. Quadrature error causing the historical HSL event: **not demonstrated**.

## 3. Newton convergence and admissibility

The prior solver audit remains applicable to the inspected source. `NSolver.solvenonlinear` (`dependencies/nsolver.py:285–346`) calls DOLFIN `solve` with the supplied UFL Jacobian in ordinary mode. Source defaults are absolute/relative tolerance 1e-7 and 50 iterations; the candidate input specifies 1e-8 and 100. The ordinary call does not explicitly select MUMPS, a preconditioner, damping, line search or a trust region. Printed “LU solver” and commented MUMPS examples are not runtime proof. Original image defaults and submitted overrides remain unauthenticated.

The current mixed Function is the next step's initial guess. Jacobians are supplied by source UFL differentiation; the upstream Newton semantics reviewed previously rebuild the Jacobian each iteration and test residual convergence, but original runtime settings still need collection. The code catches RuntimeError across ranks, flushes completed outputs and raises. It does not show an automatic step-size retry or physical fallback.

No ordinary acceptance test requires J>0, minimum singular value, bounded distortion, bounded correction, bounded inter-step deformation, good tangent conditioning, local residual distribution or material stability. The optional `MechanicalHealthDiagnostic` samples/warns after a solve; it does not enforce those conditions and is not a whole-element positivity certificate. Absence of safeguards is not itself proof they were needed or that Newton failed.

### Controlled reduced-mode experiment

The source active law is deformation-dependent. For fixed bound population total m and first moment, write

`S_cb(lambda)=a+b*lambda`, with positive b when m is nonnegative. A positive scalar dS/dlambda does not guarantee a positive tensor tangent: compressive S_cb supplies a geometric shear term.

For the 2833 event, the pre-reset scalar is conditionally reconstructed from same-stage aggregate moments as

`S_cb = A_saved + rho*k_cb*1e-9*c*m*(HSL-HSL_old) ≈ -219.8371 Pa`.

The moment representative uses two nonnegative candidate bins with the required m and mean x. **It is not the historical distribution.** Only m and its first moment enter the frozen FE CB law. Exact historical moment validity still requires correct stage mapping, unit fibers and authenticated density/stiffness/compliance.

At synthetic `F0=diag(lambda,lambda^-1/2,lambda^-1/2)` with lambda≈.390339, apply `F(gamma)=F0+gamma*(e2 tensor e1)`. J stays exactly one. The UFL tangent along this isochoric shear is:

| Contribution / check | Pa |
|---|---:|
| Passive ground shear coefficient | +125.7838 |
| Xi and fiber-collagen contribution while fiber stays compressed | 0 |
| Active coefficient | -219.8371 |
| Total UFL rank-one coefficient | **-94.0533** |
| Independent central-difference coefficient | -94.0533 |

Sheet/normal collagen invariants do not change along this reference-fiber-gradient shear. This is a precise route to local softness despite positive volume and substantial transverse collagen stiffness. It does not require negative populations or negative scalar axial stiffness.

The reduced source-law shear equation has zero residual at gamma=0 but negative derivative there. Nonzero roots occur at ±.1298081822 with positive reduced derivative ≈183.31 Pa. Unit-relaxation Newton from ±.1 reaches the corresponding nonzero root in five iterations; relaxation .5 takes 31. Starting exactly at zero passes the residual test immediately at the unstable root. Maximum unit-relaxation modal correction is about .05128. This is a scalar constrained-mode calculation with normal deformation prescribed, **not the full DOLFIN pressure/displacement/Real system or its Newton history**.

It demonstrates why residual convergence alone cannot select a mechanically stable branch. It does not prove historical branch switching, a global negative eigenvalue or element inversion. Boundary continuity and the full mesh can constrain this mode. All three reduced states have J=1.

The nonzero roots increase fiber stretch in this particular test from about
.39034 to .4114. They do **not** reproduce the original collapse from a healthier
length to .20137. A loss of rank-one stiffness can be a mechanical instability
of the modeled constitutive/active state, rather than a Newton programming
error; distinguishing that from a discretization artifact requires the original
state and controlled mesh/integration comparisons.

For synthetic isochoric geometries constructed from the other event stretches/moments, tested shear coefficients are positive (about 2384, 36234 and 40352 Pa for 4844,1580,2497). A single negative-direction test does not explain all four lengthening events.

## 4. Original deformation recovery and mapping

The prior Windows integrity evidence remains available: `solution.xdmf` is truncated at 104857600 bytes and `solution.h5` fails signature/open checks. These observations are about the Windows copy, **not the original LCC files**. No new LCC connection, full-order checkpoint, original event log or passing 32-rank suite became available in this phase.

The existing read-only collection guide is copied unchanged into this bundle. It requests original file sizes, hashes, XML/HDF5 open/group checks, source/input/runtime provenance, state inventory and latest mapping/scheduler reports before considering reruns. No originals are repaired, and no job is submitted here.

The candidate mesh's geometric point cloud matches all saved coordinates. That does not establish scalar global IDs, mixed MyoSim packing, vector fiber components, owner indices or CSV rank scatter. Use the existing scalar/mixed/geometry/ownership/scatter workflow rather than replacing it. The remote branch still contains the b5d051a launcher repair; without LCC reports its latest execution status is unresolved. A fixed launcher is not a passing mapping test.

**Exact hotspot F/J/SVD reconstruction is blocked.** Vertex-only or projected visualization cannot supply missing CG2 edge modes. The earlier quadratic edge-mode witness remains valid. Even an intact ordinary XDMF file must be checked for Function-space/dofmap/checkpoint data before using it as full-order state. Hotspot IDs and cross-field interpretations remain provisional until authenticated mapping and producer evidence are obtained.

## 5. Pressure formulation and conditional volume test

The current source still differentiates `Psi_p=-p*(J-1)` (`forms.py:268–273`, `mesh.py:1005–1009`). For the no-growth baseline this gives `S_p=-p*J*C^-1`. The manual tensor (`forms.py:359–361`) and CSV helper (`fenics_2017_compat.py:71–80`) use `-p*C^-1` and its reference-fiber contraction. The equilibrium energy derivative is correct for the stated constraint; the recorded pressure expression omits J.

This is not globally “output only”: manual total-stress expressions feed growth/fibrosis/reorientation pathways if enabled, and total-stress diagnostics inherit the discrepancy. In the reported healthy no-growth/no-fibrosis/no-reorientation baseline, the immediate MyoSim passive feedback uses Xi through its DG1 transfer, not this pressure scalar. Circulation uses cavity pressure, not manual local `incomp_stress`. No direct pressure-output feedback into this baseline equilibrium was established. Preserve this distinction when proposing any repair.

If original J and the actual saved definition are known at the **same quadrature point and stage**, the equilibrium pressure fiber contribution is `J*S_saved`. This cannot be applied blindly to DG-smoothed, differently projected, mis-mapped or rounded historical arrays. The safest calculation uses original p and F. Relative saved-versus-equilibrium difference is `1/J-1`: about +5.26% at J=.95, -4.76% at J=1.05, +25% at J=.8. No J value is measured from the pressure CSV alone.

### New conditional determinant bound from ECM and stretch

For unit f0, positive candidate fractions/moduli and the source tension-only collagen, ECM fiber stress E satisfies

`E >= G = phi_g*a_g*exp[b_g*(I1-3)]`.

Hence `I1 <= 3+log(E/(phi_g*a_g))/b_g`. For any F and unit f0 with `lambda=||F*f0||`,

`|det F| <= lambda*(I1-lambda^2)/2`.

This follows from the Schur complement and arithmetic–geometric mean on the transverse block of `C=F^T F`; it does not assume a diagonal original F. The inequality passed 200 random-matrix fixtures in the verification tests.

At 2833's minimum-stretch row, archived E≈90.54237 Pa and lambda≈.201370454. Candidate `phi_g*a_g=96.6398 Pa`, `b_g=.1158` give `I1<=2.43720` and **|J|<=.241307**. Conversely J=1 with that stretch requires at least about **216.6765 Pa** of ground fiber PK2 stress alone. Positive collagen cannot lower that bound.

This is a **conditional inconsistency with local incompressibility**, not an independent reconstruction. Before interpreting it as local volume collapse, authenticate: actual a_g/b_g/fractions; unit fibers; same mechanical state and point; direct ECM output definition; and the producer's kinematic/output implementation. A failed premise can explain the contradiction. It does not distinguish positive from negative determinant. `conditional_J_bounds_2833.csv` retains the native window.

The weak CG1 pressure equations constrain weighted moments of J−1 with the chosen quadrature, not J at every material point. Satisfying the discrete pressure residual therefore does not logically exclude large local determinant variations. The original full-order state is needed to determine whether that happened.

There are 442 candidate CG1 pressure DOFs and 5008 degree-2 material points;
global shared pressure tests do not impose 5008 independent pointwise volume
constraints. This count is not an inf-sup failure proof or a count of arbitrary
realizable J modes. Also, `LV_simulation.py:1239–1242` computes the printed
`myocardium_vol` as reference `integral 1 dx` on the unchanged mesh, not current
`integral J dx` in this no-growth path. A constant value of that diagnostic does
not verify current myocardial volume. Smooth cavity volume/pressure likewise
does not certify local J.

## 6. Passive stiffness and local stability

Xi is inactive in compression, but the full passive model is not. Ground shear remains positive; transverse expansion activates sheet/normal collagen. In the synthetic compressed state, the sampled passive isochoric acoustic coefficients range from roughly 108 Pa to 6.79e5 Pa. In the distorted state they range from roughly 137 Pa to 3.80e9 Pa. This is extreme directional contrast, not zero resistance.

Those figures use five sampled reference directions, not a complete minimum over the sphere or a global coercivity proof. Negative eigenvalues of the unconstrained local **mixed** matrix are not used as instability evidence: pressure saddle-point structure and unconstrained modes already affect its spectrum. The demonstrated negative mode instead uses an explicit admissible isochoric rank-one direction and the source tangent.

### Conditional historical rank-one certificate

Assume an orthonormal reference basis, lambda_f<1, correct scalar/stage correspondence and the inspected frozen-population active law. Choose `delta F=a tensor f0` with unit a perpendicular to both `F*f0` and `F^-T*f0`; such a direction exists for nonsingular F. This variation preserves determinant exactly, changes lambda only at second order, leaves sheet/normal family invariants unchanged, and keeps fiber Xi/collagen inactive for sufficiently small perturbations. The rank-one coefficient becomes

`Q = G + S_cb(pre-reset)`.

Under the same basis assumptions the saved ECM fiber scalar equals G in this compressed branch. Using the conditional next-row CB stage proxy from the previous report gives negative Q during **8.688–8.697 s**, minimum about **-425.96 Pa**; at 8.696 s it is about **-128.36 Pa**. Negative overlap starts later, at 8.699 s.

This is stronger than a generic conjecture: it is a source-law certificate under explicit premises. It is not an authenticated historical tangent evaluation. Nonorthogonal fibers can add collagen response to this direction; incorrect stage/point mappings or a different producer can invalidate the calculation. Nor does a local rank-one loss prove the finite global mesh actually switched branches. It identifies the smallest tangent check to perform on the recovered original state.

## 7. FE–MyoSim coupling assessment

The source active tensor is PK2 `S_cb*(f0 tensor f0)` and contributes `F*S_active` to virtual work (`mesh.py:1086–1097,1133`). Attached populations and calcium/regulation can change the next frozen coefficient state. HSL also changes the CB extension term *inside* Newton. No explicit time regularization limits the resulting mechanical change.

The prior timing ledger is preserved: cellular updates use lagged FE increment and cached calcium/stress; pre-reset CB is sampled after the solve; hsl_old is then reset; saved active stress is evaluated later. The saved active CSV is therefore not substituted for the equilibrium stress in these tests. The two-bin vector used to realize the aggregate moments is explicitly synthetic and never labeled a historical distribution.

Fixed-state tests halving synthetic attached mass from .005 to .0025 halve the active residual contribution and leave the pressure constraint residual unchanged at the frozen F. They do not determine the direction of the next displacement. In particular, reducing a **negative** active scalar can remove a destabilizing shear contribution; reducing a positive tensile scalar can cause a different redistribution. “Active-stress reduction causes shortening” is not a universal inference.

The actual source may therefore support a sequence in which initial local shortening changes CB prestress, a negative active shear coefficient overwhelms weak compressed-fiber matrix shear, and later advection/overlap changes alter the rebound. The tests demonstrate the local mechanism, not its historical trigger. They do not justify clipping negative stress, smoothing Xi or changing coupling before the original state is examined.

## 8. Hypothesis ranking

| Mechanism | Supporting evidence | Contradicting/limiting evidence | Missing evidence | Verification test | Confidence |
|---|---|---|---|---|---|
| Local active-prestress loss of rank-one stiffness | Source UFL negative mode; reduced multiple roots; conditional archived Q<0 before negative overlap | Initial mild shortening begins earlier; other three event fixtures have positive tested coefficient; full mesh may constrain mode | Original F, basis, moments, tangent and source identity | Evaluate explicit rank-one tangent at original state with pre-reset populations/reference | High for controlled mechanism; conditional for historical involvement |
| Local J departure / under-resolved constraint | Conditional ECM/stretch bound ≤.2413; weak pressure constraint; integration error observed | Bound depends on parameters, normalized fibers and point/output validity | Original p/F/J and material fields | Direct CG2 J plus higher-rule kinematic constraint residual | Strong conditional inconsistency; historical measurement blocked |
| Quadrature-related trajectory sensitivity | 0.13% residual /1.5% volume-tangent and ~0.96% cavity-tangent fixture errors | High rules converge; tested AD Jacobian is consistent; no historical trajectory comparison | Authenticated coefficients and full state | Frozen-state production-kernel analysis with justified higher-point coefficient representation | Demonstrated synthetic error; causal confidence low |
| Mapping/recording artifact | MPI/component verification incomplete; active and pressure output distinctions confirmed | Candidate geometric point cloud matches exactly; scalar algebra is internally consistent | Complete actual-rank suite and producer | Existing mapping/ownership/scatter validators, reference coordinates and fiber norms | Still a necessary exclusion |
| Newton acceptance of unstable state | No admissibility/stability acceptance; reduced Newton accepts negative-curvature root | No original iteration/correction history; residual success alone not failure | Event Newton log and full corrections | Inspect event logs; recover exact mixed state and tangent | Demonstrated general/reduced possibility, historical status unresolved |
| Xi absence alone | No Xi compression stiffness | Positive ground resistance and transverse collagen remain | Full event tangent and geometry | Constituent-resolved admissible-direction tangent | Insufficient as a standalone explanation |
| Negative overlap as initiator | Downstream kinetic guard demonstrably acts | Chronology places it after shortening and after conditional stiffness loss | None supporting initiation | Retain historical ordering; examine only amplification | Weakened as initiating cause |

## 9. Minimum proposed instrumentation — not implemented in production

At an approved targeted restart, record before/after each Newton solve and before hsl_old reset:

- Full original CG2 displacement/CG1 pressure Functions and authoritative topology/dof maps at a handful of event-adjacent steps.
- Pre-reset hsl_old, actual population vector/moments and active stress, consumed calcium/stress feedback, reference fiber norms/orthogonality and material coefficients.
- J, minimum singular value, raw stretch and selected rank-one acoustic coefficients at hotspots and face neighbors.
- Energy-consistent and legacy pressure diagnostics under distinct approved names.
- Per-iteration residual/correction norms and extrema, with physical block scaling; no automatic clipping or solver changes.
- Kinematic constraint residual evaluated independently at degrees 2/4/6, without pretending historical Quadrature fields exist at new points.

No production quadrature, solver, output format, constitutive law, coupling or parameter was changed here. Such instrumentation can change outputs/timing and requires review before use in production.

## 10. Minimal next experimental plan

| Experiment / exact objective | Required data | Existing files/functions | Cost | Discriminating evidence | Historical checkpoint needed? |
|---|---|---|---|---|---|
| Recover original integrity/provenance and latest mapping status | LCC run and job paths | Reused `LCC_Readonly_Commands.md`, ten-report mapping workflow | Storage I/O; no solve | Separates damaged copy/stale tooling from original data; authenticates parameters/basis | No |
| Test conditional volume contradiction | One original full-order u, basis, a_g/b_g/fractions and scalar correspondence near 8.707 | `Forms.Fmat/J`, existing geometry tools | Small frozen-state evaluation | Direct J confirms collapse or exposes failed output/parameter/mapping premise | Original mechanical state, not necessarily full coupled restart |
| Test rank-one loss at original onset state | Pre-reset u, hsl_old, population moments and basis near 8.688–8.697 | `MeshClass.return_cb_stress`, source UFL residual/Jacobian | Small point/cell evaluations | Actual Q sign; contributions from all families, pressure and active tangent | Original mechanical/cellular state for exact evaluation |
| Isolate pressure-constraint integration error | Original u only plus CG1 test space | Pure `q*(det(I+grad(u))-1)` diagnostic | Small element/global assembly, no solve | Degree-2/4/6 discrepancy and localized J residual | Original u required for historical claim |
| Compare complete frozen residual/tangent consistently | Original mixed state plus a justified extension of every Quadrature coefficient | Production forms and existing mapping | Moderate; requires FE runtime | Degree sensitivity in the actual assembled operator | Yes, coefficient state too |
| Controlled short restart, only after approval | Full coupled checkpoint before event | Unchanged production equations/settings plus approved recorder | Tens/hundreds of steps; setup/I/O dependent | Reproduces or rejects actual branch/feedback sequence | Full coupled checkpoint essential |

Without a full coupled checkpoint, CSVs cannot restart the event. The prior cost estimate and caution against reintegrating from time zero are retained. No expensive job or LV replay was launched.

## Executed tests and limitations

- **69 focused numerical checks passed:** FIAT monomial exactness through each tested rule degree; homogeneous-reference tangent control; degree-10/12 reference convergence; fixed-state constraint invariance under active-mass perturbation; actual UFL acoustic-versus-FD checks for four event fixtures; negative 2833 fixture direction; six reduced Newton runs; determinant inequality controls; source UFL/nodal FD consistency; cavity convergence; per-form FFC metadata; candidate geometry/basis checks.
- Four actual UFL source-law derivative checks and 12 nodal finite-difference checks are tabulated separately. FFC analyzed all ten requested residual/Jacobian categories successfully with degree-2 metadata.
- **92 existing local diagnostic tests passed** in 17.922 s; see `existing_tests.txt`. No real DOLFIN, 1-rank FE, 32-rank MPI, original-state or historical-replay test was executed.
- Independent local assembly does not validate distributed ordering, full LV boundary conditions or the actual compiled production kernels. Numerical convergence of these synthetic tests cannot authenticate the original run.
- The pressure-output discrepancy is not itself a residual formulation error. The local instability test is not proof that a production global tangent eigenvalue was negative. The conditional scalar checks are not measured J or recovered original populations.

Reproducible scripts, generated numerical source-law evaluator, configurations/provenance hashes, convergence/error CSVs, reduced Newton traces and plots are included in the evidence bundle. Original files and prior reports were preserved.
