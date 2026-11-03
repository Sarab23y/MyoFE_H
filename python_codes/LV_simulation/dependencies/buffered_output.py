"""Dependency-light invariants for bounded output buffering (Python 2.7)."""
from __future__ import division

import numbers


def is_scalar_output_value(value):
    """Return True only for values which fit one numeric CSV cell.

    ``np.isscalar`` alone is not sufficient for zero-dimensional arrays and
    accepting one-element vectors is dangerous: those vectors can later grow
    into material-point fields without changing their registration path.
    """
    if isinstance(value, numbers.Number):
        return True
    # NumPy scalars implement ``numbers.Number``.  A zero-dimensional ndarray
    # is accepted only when its dtype kind is numeric; use duck typing here so
    # this small helper remains importable in dependency-light validators.
    return (getattr(value, 'ndim', None) == 0 and
            getattr(getattr(value, 'dtype', None), 'kind', '') in
            'biufc')


def scalar_field_names(*mappings):
    """Classify global-output fields from their current values."""
    names = []
    for mapping in mappings:
        for name, value in mapping.items():
            if name not in names and is_scalar_output_value(value):
                names.append(name)
    return names


class BoundedRowBuffer(object):
    """Small reusable row buffer used by unit tests and output adapters."""
    def __init__(self, fields, capacity):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.fields = tuple(fields); self.capacity = int(capacity)
        self.rows = []

    def append(self, row):
        unknown = set(row)-set(self.fields)
        missing = set(self.fields)-set(row)
        if unknown or missing:
            raise ValueError('row fields differ from configured fields')
        self.rows.append(tuple(row[name] for name in self.fields))
        return len(self.rows) == self.capacity

    def drain(self):
        rows = self.rows; self.rows = []
        return rows


def validate_global_dof_partition(rank_dofs, point_count):
    """Reject missing, duplicate, negative, or out-of-range owned DOFs."""
    flat = [int(value) for values in rank_dofs for value in values]
    return (len(flat) == point_count and len(set(flat)) == point_count and
            sorted(flat) == list(range(point_count)))


def estimated_numeric_buffer_bytes(field_count, point_count, batch_size,
                                   scalar_field_count=0):
    """Lower-bound estimate for float64 numeric payload storage."""
    return 8*batch_size*(field_count*point_count + scalar_field_count)

