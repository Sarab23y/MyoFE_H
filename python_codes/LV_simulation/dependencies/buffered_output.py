"""Dependency-light invariants for bounded output buffering (Python 2.7)."""
from __future__ import division


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

