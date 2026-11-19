"""Common read-only measurement arm; no F/event/population-stage instrumentation.

This is distinguished from a pure disabled arm. Both arms use the original solve.
"""
from __future__ import division
import os
import numpy as np
from .fe_cycle_diagnostic import FECycleDiagnostic, clean_json
from .native_newton_telemetry import NewtonTelemetry
from .fe_cycle_core import CycleLedger


class InvarianceWitness(FECycleDiagnostic):
    def __init__(self,parent,config):
        self.parent=parent;self.config=config;self.comm=parent.comm;self.rank=self.comm.Get_rank()
        self.root=os.path.join(os.path.dirname(config['directory']),'witness')
        self.log_error=None;self.stop=False;self.closed=False
        self.ledger=CycleLedger(config['complete_cycles'])
        self.functions=parent.mesh.model['functions'];self.w=self.functions['w']
        self.q=parent.mesh.model['function_spaces']['quadrature_space']
        def create():
            if self.rank==0:os.makedirs(self.root)
        self.local_io(create)
        self.log=self.local_io(lambda:open(os.path.join(self.root,'rank%04d.jsonl'%self.rank),'w'))
        self.newton=NewtonTelemetry(lambda:self.w.sub(0,deepcopy=True).vector().get_local(),self.emit,self.comm)

    def begin(self):
        self.step=int(self.parent.t_counter);self.label=float(self.parent.data['time'])
        self.accepted_time=self.label+float(self.parent.prot.data['time_step'])

    def before_solve(self):
        self.sync_log();self.newton.start()

    def accepted(self):
        self.newton_result=self.newton.finish(True)
        self.equilibrium=self.project(self.functions['cb_stress'],'witness_equilibrium_active')

    def end(self):
        capture_step(self,self.root,self.equilibrium,self.newton_result)
        self.ledger.observe(self.accepted_time,self.parent.data['new_beat'])
        self.stop=self.step+1>=int(self.config['smoke_steps'])

    def finish(self,success=False,error=None):
        if self.closed:return
        if self.newton.active:
            try:self.newton.finish(False)
            except RuntimeError as diagnostic_error:
                if error is None:raise
                self.emit('telemetry_error_during_original_failure',dict(original_error=error,telemetry_error=str(diagnostic_error)))
        self.emit('witness_finished',dict(success=success,error=error,pure_disabled_arm=False))
        self.sync_log();self.log.close();self.newton.close();self.closed=True


def capture_step(observer,directory,equilibrium,newton):
    """Same per-step evidence in full and witness arms, bounded on-disk files."""
    import json
    parent=observer.parent
    payload=dict(global_ids=np.asarray(parent.dofmap),
        mixed_owned=observer.w.vector().get_local().copy(),
        legacy_time=observer.label,accepted_time=observer.accepted_time,
        hsl=np.asarray(parent.hs_length_list),stretch=np.asarray(parent.data['fiber_stretch_raw']),
        calcium=np.asarray([hs.memb.data['Ca_cytosol'] for hs in parent.hs_objs_list]),
        y_cellular=np.asarray([hs.myof.y for hs in parent.hs_objs_list]),
        y_FE=observer.functions['y_vec'].vector().get_local().copy(),
        active_equilibrium=np.asarray(equilibrium),
        newton_json=np.asarray(json.dumps(clean_json(newton),allow_nan=False)))
    path=os.path.join(directory,'physical_%08d_rank%04d.npz'%(observer.step,observer.rank))
    def write():
        descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(descriptor,'wb') as handle:np.savez_compressed(handle,**payload)
    observer.local_io(write)
