"""Report and root-rank error handling for standalone/package diagnostics."""
from __future__ import print_function
import json
import os
import sys

PYTHON2 = sys.version_info[0] == 2
try:
    TEXT_TYPE = unicode
except NameError:
    TEXT_TYPE = str


def native_filename(path):
    """Match production's UTF-8 recoding for Python-2 SWIG std::string input.

    Python 3 bindings receive native text, never the repr of a bytes object.
    Conversion changes the Python representation, not the selected file.
    """
    if hasattr(path, '__fspath__'):
        path = path.__fspath__()
    if isinstance(path, TEXT_TYPE):
        result = path.encode('utf-8') if PYTHON2 else str(path)
    elif isinstance(path, bytes):
        result = path if PYTHON2 else path.decode('utf-8')
    else:
        raise TypeError('HDF5 filename must be text/bytes/path-like, got %s' % type(path).__name__)
    if not result or (b'\x00' if isinstance(result, bytes) else '\x00') in result:
        raise ValueError('HDF5 filename is empty or contains a null byte')
    return result


def open_hdf5_file(constructor, comm, path, mode='r'):
    """Use one typed filename boundary for every diagnostic DOLFIN HDF read."""
    filename = native_filename(path)
    rank = None
    for method in ('Get_rank', 'getRank'):
        query = getattr(comm, method, None)
        if callable(query):
            rank = query()
            break
    detail = dict(python=sys.version.split()[0], original_filename_type=type(path).__name__,
                  filename_type=type(filename).__name__, filename_repr=repr(filename),
                  communicator_type=type(comm).__module__+'.'+type(comm).__name__,
                  mode=mode, signature='HDF5File(comm, filename, mode)')
    if rank in (0, None):
        print('[Diagnostic HDF5 constructor] '+json.dumps(detail, sort_keys=True))
        sys.stdout.flush()
    try:
        return constructor(comm, filename, mode)
    except Exception as error:
        raise RuntimeError('Diagnostic HDF5 constructor failed: %s; arguments=%s' %
                           (error, json.dumps(detail, sort_keys=True)))


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
