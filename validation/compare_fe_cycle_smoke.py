"""Streaming output comparison; never turns partial FE evidence into readiness."""
from __future__ import print_function
import argparse
import csv
import glob
import itertools
import json
import os
import numpy as np


def compare_csv(left, right, atol=1.e-10, rtol=1.e-9, expected_dt=None):
    rows = 0
    maximum = 0.
    previous = None
    first = None
    with open(left) as a, open(right) as b:
        ar, br = csv.reader(a), csv.reader(b)
        header = next(ar)
        if header != next(br):
            raise ValueError('CSV headers differ: '+left)
        if header.count('time') != 1:
            raise ValueError('missing/ambiguous explicit time column: '+left)
        time_column = header.index('time')
        for x,y in itertools.zip_longest(ar,br):
            if x is None or y is None:
                raise ValueError('CSV row counts differ')
            xx,yy = np.asarray(x,dtype=float), np.asarray(y,dtype=float)
            if not np.allclose(xx,yy,atol=atol,rtol=rtol,equal_nan=False):
                raise ValueError('CSV numeric difference at row %d: %s'%(rows,left))
            current=float(xx[time_column])
            if previous is not None:
                if current <= previous:
                    raise ValueError('duplicate or nonincreasing timestamp: '+left)
                if expected_dt is not None and abs(current-previous-expected_dt)>1.e-9:
                    raise ValueError('missing/off-grid timestamp: '+left)
            else:
                first=current
            previous=current
            maximum=max(maximum,float(np.max(np.abs(xx-yy))))
            rows += 1
    if not rows:
        raise ValueError('empty CSV')
    return dict(rows=rows,max_abs_difference=maximum,first_time=first,last_time=previous)


def compare(left,right,pure=None):
    files = sorted(glob.glob(os.path.join(left,'sim_output','*.csv')))
    results = {'initial_state':compare_initial(left,right)}
    for path in files:
        name = os.path.basename(path)
        results[name] = compare_csv(path,os.path.join(right,'sim_output',name),
                                   expected_dt=.001 if name=='data.csv' else .05)
    if 'data.csv' not in results or results['data.csv']['rows'] != 200:
        raise ValueError('smoke must contain exactly 200 native global rows')
    if abs(results['data.csv']['first_time'])>1.e-9 or abs(results['data.csv']['last_time']-.199)>1.e-9:
        raise ValueError('smoke global timestamps must cover legacy labels 0 through 0.199 s')
    results['final_vectors']=compare_final(left,right)
    results['per_step_and_native_Newton']=compare_native(left,right)
    if pure:
        results['pure_disabled_control']=dict(initial_state=compare_initial(pure,right),final_vectors=compare_final(pure,right),csv={})
        for path in files:
            name=os.path.basename(path)
            results['pure_disabled_control']['csv'][name]=compare_csv(os.path.join(pure,'sim_output',name),os.path.join(right,'sim_output',name),expected_dt=.001 if name=='data.csv' else .05)
    return dict(physical_outputs_compared_passed=True,checks=results,
                full_pipeline_validated=False,newton_trial_invariance='native counts/residuals/corrections compared; witness has no trial F projections')


def compare_initial(left,right):
    files=sorted(glob.glob(os.path.join(left,'initial_rank*.npz')))
    names=[os.path.basename(p) for p in files]
    if not files or names!=sorted(os.path.basename(p) for p in glob.glob(os.path.join(right,'initial_rank*.npz'))):
        raise ValueError('initial-state witness rank sets missing/different')
    for path in files:
        with np.load(path,allow_pickle=False) as a,np.load(os.path.join(right,os.path.basename(path)),allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ValueError('initial source field sets differ')
            for key in a.files:
                if not np.array_equal(a[key],b[key]):
                    raise ValueError('initial states differ before stepping; not an instrumentation-solution comparison: '+key)
    return dict(exact_initial_evidence_equal=True,rank_files=len(files),full_coupled_restart=False)


def compare_final(left,right):
    results={}
    final = sorted(glob.glob(os.path.join(left,'final_rank*.npz')))
    if not final:
        raise ValueError('no final displacement coefficient captures')
    for path in final:
        other = os.path.join(right,os.path.basename(path))
        maxima={}
        with np.load(path) as a, np.load(other) as b:
            if set(a.files) != set(b.files):
                raise ValueError('final fields differ')
            for name in a.files:
                maxima[name]=float(np.max(np.abs(a[name]-b[name]))) if a[name].size else 0.
                if not np.allclose(a[name],b[name],atol=1.e-10,rtol=1.e-9,equal_nan=False):
                    raise ValueError('final coefficient/state differs: '+name)
        results[os.path.basename(path)] = dict(max_abs_difference=maxima,atol=1.e-10,rtol=1.e-9)
    if len(final) != len(glob.glob(os.path.join(right,'final_rank*.npz'))):
        raise ValueError('rank sets differ')
    return results


def compare_native(left,right):
    """Compare every accepted step, not just final state and decimated CSVs."""
    a_files=sorted(glob.glob(os.path.join(left,'witness','physical_*.npz')))
    b_files=sorted(glob.glob(os.path.join(right,'observer','physical_*.npz')))
    if not a_files or [os.path.basename(p) for p in a_files]!=[os.path.basename(p) for p in b_files]:
        raise ValueError('missing/mismatched native per-step witness frames')
    ranks=len(glob.glob(os.path.join(left,'final_rank*.npz')))
    expected=set('physical_%08d_rank%04d.npz'%(step,rank) for step in range(200) for rank in range(ranks))
    if set(os.path.basename(p) for p in a_files)!=expected:
        raise ValueError('incomplete native 200-step/rank grid')
    differences={};maxima={}
    for a_path,b_path in zip(a_files,b_files):
        with np.load(a_path,allow_pickle=False) as a,np.load(b_path,allow_pickle=False) as b:
            if set(a.files)!=set(b.files):raise ValueError('native state fields differ')
            for key in a.files:
                if key=='newton_json':continue
                if a[key].shape!=b[key].shape:raise ValueError('native state shape mismatch: '+key)
                delta=float(np.max(np.abs(a[key]-b[key]))) if a[key].size else 0.
                maxima[key]=max(maxima.get(key,0.),delta)
                if not np.allclose(a[key],b[key],atol=1.e-10,rtol=1.e-9,equal_nan=False):
                    raise ValueError('native timestep physical difference: %s in %s'%(key,a_path))
            an=json.loads(str(a['newton_json']));bn=json.loads(str(b['newton_json']))
            for key in ('iterations','status','residual_evaluations'):
                if an[key]!=bn[key]:raise ValueError('Newton convergence/count mismatch: '+key)
            if len(an['records'])!=len(bn['records']):raise ValueError('Newton residual history lengths differ')
            for av,bv in zip(an['records'],bn['records']):
                for key in ('iteration','backend_converged','cumulative_linear_iterations'):
                    if av[key]!=bv[key]:raise ValueError('native Newton metadata differs: '+key)
                for key in ('residual_l2','relative_residual'):
                    aa,bb=av[key],bv[key]
                    if aa is None or bb is None:
                        if aa!=bb:raise ValueError('native residual validity differs')
                    elif not np.isclose(aa,bb,atol=1.e-12,rtol=1.e-9):
                        raise ValueError('native Newton residual differs: '+key)
                ac,bc=av['correction'],bv['correction']
                if (ac is None)!=(bc is None):raise ValueError('missing native correction')
                if ac:
                    for key in ac:
                        if ac[key] is None or bc[key] is None:
                            if ac[key]!=bc[key]:raise ValueError('correction validity differs')
                        elif not np.isclose(ac[key],bc[key],atol=1.e-12,rtol=1.e-9):
                            raise ValueError('native displacement correction differs: '+key)
    return dict(frames=len(a_files),max_abs_physical_difference=maxima,
        physical_atol=1.e-10,physical_rtol=1.e-9,newton_atol=1.e-12,newton_rtol=1.e-9,
        common_measurement_witness=True,pure_disabled_arm_also_recommended=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',required=True)
    p.add_argument('--enabled',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--pure-control')
    a=p.parse_args()
    result=compare(a.control,a.enabled,a.pure_control)
    descriptor=os.open(a.output,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    with os.fdopen(descriptor,'w') as handle:
        json.dump(result,handle,indent=2)


if __name__=='__main__':
    main()
