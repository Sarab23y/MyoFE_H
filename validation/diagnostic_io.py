"""Report and root-rank error handling for standalone/package diagnostics."""
from __future__ import print_function
import json
import os


def write_report(path, report):
    """Create a new report exclusively. Never replace an existing result."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
    except Exception:
        os.unlink(path)
        raise


def root_call(comm, callback):
    """All ranks receive the same root-only outcome, including I/O failures."""
    outcome = None
    if comm.Get_rank() == 0:
        try:
            outcome = (None, callback())
        except Exception as error:
            outcome = ('%s: %s' % (type(error).__name__, error), None)
    error, result = comm.bcast(outcome, root=0)
    if error is not None:
        raise RuntimeError('Diagnostic rank-zero processing failed: ' + error)
    return result
