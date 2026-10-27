"""Dependency-light scalar reporting helpers for ECM baseline stability."""
from __future__ import division
import math


def symmetric_relative_change(previous, current):
    previous, current = float(previous), float(current)
    if (math.isnan(previous) or math.isinf(previous) or
            math.isnan(current) or math.isinf(current)):
        raise ValueError('Stability values must be finite')
    scale = max(abs(previous), abs(current))
    if scale == 0.0:
        return 0.0
    return abs(current-previous)/scale


def stability_results(records, field_changes, tolerance):
    if len(records) < 2 or len(field_changes) != len(records)-1:
        raise ValueError('Stability requires matching consecutive cycle data')
    results = []
    for index, (previous, current) in enumerate(zip(records[:-1], records[1:])):
        changes = {
            'pressure': symmetric_relative_change(previous['pressure'], current['pressure']),
            'volume': symmetric_relative_change(previous['volume'], current['volume']),
            'heart_rate': symmetric_relative_change(previous['heart_rate'], current['heart_rate']),
            'ecm_stress_field': float(field_changes[index])}
        results.append({'previous_cycle': index, 'current_cycle': index+1,
                        'changes': changes,
                        'passed': dict((key, value <= tolerance)
                                       for key, value in changes.items())})
    return results
