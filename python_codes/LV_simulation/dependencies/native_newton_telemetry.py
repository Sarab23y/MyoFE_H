"""Observe original C++ Newton calls; no Python replacement Newton algorithm."""
from __future__ import division
import ctypes
import os
import time
import numpy as np

_LIVE_OBSERVER=None  # C++ holds a function pointer, not a Python owning reference.


class NewtonTelemetry(object):
    def __init__(self, displacement, emit, comm, trial_metrics=None, library=None):
        global _LIVE_OBSERVER
        if _LIVE_OBSERVER is not None:
            raise RuntimeError('another native Newton observer is already registered')
        self.displacement=displacement
        self.emit=emit
        self.comm=comm
        self.trial_metrics=trial_metrics
        self.active=False
        self.error=None
        self.records=[]
        path=library or os.environ.get('MYOFE_NEWTON_OBSERVER')
        if not path or not os.path.isfile(path):
            raise RuntimeError('missing compiled/preloaded MYOFE_NEWTON_OBSERVER')
        if os.path.realpath(path) not in [os.path.realpath(p) for p in os.environ.get('LD_PRELOAD','').split(':') if p]:
            raise RuntimeError('observer must be preloaded before Python/DOLFIN starts')
        self.handle=ctypes.CDLL(path)
        self.handle.myofe_newton_observer_abi.restype=ctypes.c_int
        if self.handle.myofe_newton_observer_abi()!=1:
            raise RuntimeError('unsupported observer ABI')
        self.callback_type=ctypes.CFUNCTYPE(None,ctypes.c_int,ctypes.c_size_t,
            ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_size_t)
        self.callback=self.callback_type(self._safe_callback)
        self.handle.myofe_newton_set_callback.argtypes=[self.callback_type]
        self.handle.myofe_newton_set_callback.restype=None
        self.handle.myofe_newton_set_callback(self.callback)
        _LIVE_OBSERVER=self

    def start(self):
        self.records=[];self.error=None;self.correction=None;self.before=None;self.active=True;self.callback_wall_s=0.

    def _safe_callback(self,*args):
        if not self.active:
            return
        errors=self.comm.allgather(self.error)
        if any(errors):
            self.error=str(errors);return
        started=time.time()
        try:self.observe(*args)
        except BaseException as exc:
            # Never unwind a Python exception through a C++ solver callback.
            self.error='%s: %s'%(type(exc).__name__,exc)
        finally:self.callback_wall_s+=time.time()-started
        errors=self.comm.allgather(self.error)
        if any(errors):self.error=str(errors)

    def local(self,callback):
        error=None;value=None
        try:value=callback()
        except BaseException as exc:error='%s: %s'%(type(exc).__name__,exc)
        errors=self.comm.allgather(error)
        if any(errors):raise RuntimeError('collective native observer error: '+str(errors))
        return value

    def observe(self,event,iteration,residual,relative,relaxation,converged,linear):
        if event==1:
            self.before=self.local(lambda:np.asarray(self.displacement()).copy())
        elif event==2:
            def correction():
                if self.before is None:raise RuntimeError('missing pre-update callback')
                delta=np.asarray(self.displacement())-self.before
                return (float(np.dot(delta,delta)),float(np.max(np.abs(delta))) if len(delta) else 0.)
            reductions=self.comm.allgather(self.local(correction))
            norm=float(np.sqrt(sum(v[0] for v in reductions)))
            maximum=max(v[1] for v in reductions)
            self.correction=dict(applied_displacement_l2=norm,applied_displacement_max=maximum,
                linear_displacement_l2=norm/abs(relaxation) if relaxation else None,
                linear_displacement_max=maximum/abs(relaxation) if relaxation else None,
                relaxation=relaxation,iteration=int(iteration))
        elif event==0:
            def check_sequence():
                if int(iteration)!=len(self.records):raise RuntimeError('missing/repeated native residual callback')
                if iteration and (self.correction is None or self.correction['iteration']!=int(iteration)):
                    raise RuntimeError('missing native displacement-update callback')
            self.local(check_sequence)
            values=dict(iteration=int(iteration),residual_l2=float(residual),
                relative_residual=float(relative),backend_converged=bool(converged),
                cumulative_linear_iterations=int(linear),state_category='trial_residual_evaluation',
                correction=self.correction.copy() if self.correction else None,
                trial=self.trial_metrics() if self.trial_metrics else None)
            self.records.append(values)
            self.emit('newton_iteration',values)
        else:raise RuntimeError('unknown native callback event')

    def finish(self,returned=True):
        self.active=False
        errors=self.comm.allgather(self.error)
        if any(errors):raise RuntimeError('Newton telemetry callback failed: '+str(errors))
        if returned and not self.records:
            raise RuntimeError('native Newton callbacks not reached')
        status=('solver_failed' if not returned else 'converged' if self.records[-1]['backend_converged']
                else 'backend_returned_nonconverged')
        result=dict(status=status,
            iterations=self.records[-1]['iteration'] if self.records else None,
            residual_evaluations=len(self.records),records=self.records,
            callback_wall_s_max_rank=max(self.comm.allgather(self.callback_wall_s)),
            callback_observed=bool(self.records))
        self.emit('newton_solve_status',result)
        return result

    def close(self):
        global _LIVE_OBSERVER
        self.active=False
        self.handle.myofe_newton_set_callback(self.callback_type())
        if _LIVE_OBSERVER is self:_LIVE_OBSERVER=None
