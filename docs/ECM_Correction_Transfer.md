# ECM correction transfer bundle

`validation/export_ecm_corrections.py` creates an allow-listed ZIP for moving
the ECM spatial-investigation corrections out of an isolated checkout. The
bundle contains complete copies of the diagnostic implementation, its small
dependency closure, driver/entry-point corrections, the CSV analyzer, focused
tests, and investigation documentation.

The bundle deliberately excludes simulation results, meshes, XDMF/HDF5 files,
plots, Git metadata, and unrelated source files. Every included file and the
cumulative patch have a SHA-256 digest in `TRANSFER_MANIFEST.json`.

The cumulative patch is generated from local baseline `daab60c`. The current
GitHub master reported by the user, `8c1f57d4a6dee30b962460eb4759e54dfbcf3a49`,
is not present in this isolated checkout. As a result, the archive instructions
require `git apply --check` and prohibit forcing a conflicting patch or blindly
overwriting the main driver.

Example export:

```bash
python validation/export_ecm_corrections.py \
  --baseline daab60c \
  --head HEAD \
  --output /workspace/MyoFE_H_ECM_corrections.zip
```

The generated `APPLY_INSTRUCTIONS.md` gives the fresh-branch, review, apply,
test, and pull-request workflow.
