"""Pure numerical helpers for the passive-ECM stress diagnostic."""
from __future__ import division
import numpy as np


def positive_excess(current, homeostatic):
    signed = np.asarray(current, dtype=float) - np.asarray(homeostatic, dtype=float)
    return signed, np.maximum(signed, 0.0)


def three_cycle_homeostatic_reference(cycle_fields):
    if len(cycle_fields) != 3:
        raise ValueError('ECM homeostasis requires exactly three cycles')
    fields = [np.asarray(field, dtype=float) for field in cycle_fields]
    if not all(field.shape == fields[0].shape for field in fields):
        raise ValueError('ECM cycle fields must have identical point ordering')
    result = np.mean(np.vstack(fields), axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError('ECM homeostatic reference must be finite')
    return result


def relative_span(values):
    values = np.asarray(values, dtype=float)
    scale = max(abs(float(np.mean(values))), np.finfo(float).eps)
    return float((np.max(values) - np.min(values))/scale)
