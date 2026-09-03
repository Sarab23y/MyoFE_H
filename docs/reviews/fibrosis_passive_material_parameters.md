# Fibrosis passive-material parameters

The maintained passive formulation separates three constituent parameter
groups under `mesh.forms_parameters.passive_law_parameters`:

- `ground_matrix`: `a_g`, `b_g`;
- `myofiber`: the existing Xi parameters `c2`, `c3`;
- `collagen`: independent `a_cf`, `b_cf`, `a_cs`, `b_cs`, `a_cn`, `b_cn`.

The values in `demos/base/sim_inputs/base_instruction.json` for the ground
matrix and collagen are reference/literature values supplied from another
work. They have not been calibrated for this Fibrosis model.

The FE mechanics use Pa: cavity pressure is multiplied by `0.0075` when it is
reported in mmHg. Therefore, stress-like literature values supplied in kPa
are stored in the JSON after multiplication by 1000. All `b_*` coefficients
are dimensionless and are not converted.

The literature myofiber values `a_m` and `b_m` are not used because they
belong to a different myofiber energy. The retained Xi law continues to use
its independently calibrated `c2` and `c3` parameters. The mixture fractions
`phi_m`, `phi_g`, and `phi_c` remain separate top-level passive parameters;
the example keeps `phi_c = 0` to avoid inventing a collagen volume fraction.
Set physically appropriate fractions before using the example as a Fibrosis
production case.
