"""Actual MPI collective/event-window test with explicitly manufactured markers."""
from __future__ import print_function
import argparse
import os
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-size',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    from mpi4py import MPI
    from diagnostic_io import root_call,write_report
    repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,os.path.join(repo,'python_codes'))
    from LV_simulation.dependencies.fe_cycle_core import EventWindow
    comm=MPI.COMM_WORLD
    if comm.Get_size()!=a.expected_size:raise RuntimeError('MPI allocation size mismatch')
    window=EventWindow(20,50);written=[]
    for step in range(131):
        local=(step==40 and comm.Get_rank()==comm.Get_size()-1) or (step==60 and comm.Get_rank()==0)
        triggered=any(comm.allgather(local))
        selected=window.add(step,dict(rank=comm.Get_rank(),step=step),triggered)
        written.extend(n for n,frame in selected)
        if len(window.history)>21:raise RuntimeError('unbounded event buffer')
    all_steps=comm.allgather(written)
    expected=list(range(20,111))
    if any(steps!=expected for steps in all_steps):raise RuntimeError('MPI event union missing/duplicate frames')
    root_call(comm,lambda:write_report(a.output,dict(passed=True,mpi_size=comm.Get_size(),
        manufactured_rank_markers=True,actual_FE_mapping_verified=False,
        nonroot_trigger_propagated=True if comm.Get_size()>1 else 'not applicable at one rank',
        expected_steps=expected,pre_ms=20,post_ms=50,bounded_frames=21)))


if __name__=='__main__':main()
