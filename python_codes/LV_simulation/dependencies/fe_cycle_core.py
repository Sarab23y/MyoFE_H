"""Read-only diagnostic rules. Python 2.7/3 compatible; no FE imports."""
from __future__ import division
from collections import deque
import numpy as np


class CycleLedger(object):
    """Thirty intervals require 31 genuine activation starts, not 30 markers."""
    def __init__(self, target):
        if int(target) != target or target < 1:
            raise ValueError('target must be a positive integer')
        self.target = int(target)
        self.start = None
        self.cycles = []
        self.last_time = None

    def observe(self, time, new_beat):
        if self.last_time is not None and time <= self.last_time:
            raise ValueError('cycle observations must have increasing timestamps')
        self.last_time = float(time)
        if new_beat:
            if self.start is not None:
                self.cycles.append(dict(cycle=len(self.cycles)+1,
                                        start=self.start, end=float(time)))
            self.start = float(time)
        return len(self.cycles) == self.target


def deformation_metrics(F, fiber, hsl0):
    F = np.asarray(F, dtype=float)
    fiber = np.asarray(fiber, dtype=float)
    singular = np.full((len(F),3), np.nan)
    valid = np.all(np.isfinite(F), axis=(1,2))
    singular[valid] = np.linalg.svd(F[valid], compute_uv=False)
    stretch = np.linalg.norm(np.einsum('nij,nj->ni', F, fiber), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        condition = singular[:, 0]/singular[:, -1]
    return dict(J=np.linalg.det(F), stretch=stretch,
                strain=0.5*(stretch**2-1), hsl=stretch*np.asarray(hsl0),
                sigma_max=singular[:, 0], sigma_mid=singular[:, 1],
                sigma_min=singular[:, 2], condition=condition,
                fiber_norm=np.linalg.norm(fiber, axis=1))


def population_metrics(y, reference_sum=None):
    """Supported SRX schemes: y[:2] detached, y[2:-2] bound, y[-2:] thin."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] < 5:
        raise ValueError('expected point by population matrix')
    myosin = np.sum(y[:, :-2], axis=1)
    return dict(M_SRX=y[:, 0], M_DRX=y[:, 1],
                bound_sum=np.sum(y[:, 2:-2], axis=1),
                detached_sum=np.sum(y[:, :2], axis=1), myosin_sum=myosin,
                myosin_conservation_error=(myosin-reference_sum
                    if reference_sum is not None else np.full(len(y), np.nan)),
                min_bin=np.min(y, axis=1), max_bin=np.max(y, axis=1),
                negative_bins=np.sum(y < 0, axis=1),
                negative_bound_bins=np.sum(y[:, 2:-2] < 0, axis=1),
                n_off=y[:, -2], n_on=y[:, -1])


def triggers(fields, previous, limits):
    n = len(fields['J'])
    masks = dict(low_stretch=fields['stretch'] < limits['stretch'],
                 low_J=fields['J'] < limits['J'], inverted=fields['J'] <= 0,
                 low_singular=fields['sigma_min'] < limits['sigma_min'])
    masks['stretch_jump'] = (np.zeros(n, dtype=bool) if previous is None else
                            np.abs(fields['stretch']-previous) > limits['delta_stretch'])
    for name in ('overlap', 'min_bin'):
        if name in fields:
            masks['negative_'+name] = fields[name] < 0
    masks['nonfinite'] = np.zeros(n, dtype=bool)
    for value in fields.values():
        value = np.asarray(value)
        if value.shape == (n,):
            masks['nonfinite'] |= ~np.isfinite(value)
    return masks


class EventWindow(object):
    """Bounded history; union overlapping events without writing a frame twice."""
    def __init__(self, before=20, after=50):
        self.history = deque(maxlen=int(before)+1)
        self.after = int(after)
        self.until = -1
        self.last_written = -1

    def add(self, step, frame, triggered):
        self.history.append((step, frame))
        if triggered:
            self.until = max(self.until, step+self.after)
        selected = []
        if step <= self.until:
            for number, value in self.history:
                if number > self.last_written:
                    selected.append((number, value))
                    self.last_written = number
        return selected

    def unfinished(self, last_step):
        return max(0, self.until-last_step)
