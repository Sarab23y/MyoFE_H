# DG1 stress-pathway audit

| Path | Source / measure | Transfer | Destination | Physical effect | ECM provisional case |
|---|---|---|---|---|---|
| MyoSim passive feedback | Xi fiber PK2 scalar `Sff` | DG1 projection, then interpolation | scalar quadrature `pass_stress_list` | **Yes**, MyoSim input; deliberately unchanged | Enabled |
| Growth signal | total or Xi PK2 fiber signal | total stress uses `growth_scalar_FS` projection then quadrature interpolation | growth component arrays | Only when growth enabled | Disabled |
| Fibrosis signal | total PK2 fiber stress | `growth_scalar_FS`/DG1 projection then quadrature interpolation when available | fibrosis cycle history | Only when fibrosis enabled | Disabled |
| Fiber reorientation | active and passive PK2 traction directions | vector DG1 projection | fiber update space | Only when reorientation enabled | Disabled |
| Historical stress CSV | fiber contractions of passive tensors | releases before direct-quadrature correction could mix DG1 and quadrature ordering | CSV columns labeled as quadrature DOFs | Recording only | Historical files require caution |
| ECM-only diagnostic | fraction-weighted ground+collagen PK2 fiber stress | **direct degree-2 Quadrature projection** | ECM cycle history | Diagnostic only | Enabled |
| XDMF visualization | requested scalar/tensor expressions | projection to visualization scalar/DG spaces | XDMF | Recording only | Enabled for requested fields |

DG1 smoothing is not pointwise equivalent to evaluation at material
quadrature points.  It is not globally replaced here: the MyoSim coupling,
growth, fibrosis and reorientation pathways retain their established
behavior.  The standalone MPI validator quantifies DG1/direct differences
and tests direct-field agreement by physical quadrature coordinates across
one and 32 ranks.
