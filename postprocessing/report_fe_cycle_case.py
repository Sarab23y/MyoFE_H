"""Bounded-frame postprocessing with explicit evidence limits; Python 3 + NumPy."""
import argparse
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np


def report(instruction):
    with open(instruction) as handle:
        case=json.load(handle)
    directory=case['fe_cycle_diagnostic']['directory']
    output=os.path.join(os.path.dirname(directory),'analysis')
    os.mkdir(output)
    rows=[]
    events=[]
    event_count=0
    solves=[]
    with open(os.path.join(directory,'rank0000.jsonl')) as handle:
        for line in handle:
            item=json.loads(line)
            if item['kind']=='global':
                v=item['values'];m=v['mechanics']
                rows.append(dict(time=item['accepted_time'],legacy_time=item['legacy_time'],
                    pressure=v['pressures']['pressure_ventricle'],
                    volume=v['volumes']['volume_ventricle'],heart_rate=v['heart_rate'],**m))
            if item['kind']=='solve_returned':
                solves.append(dict(time=item['accepted_time'],**item['values']))
    # Stream all ranks' event rows, including events confined to non-root ranks.
    with open(os.path.join(output,'events.jsonl'),'w') as target:
        for path in sorted(glob.glob(os.path.join(directory,'rank*.jsonl'))):
            with open(path) as handle:
                for line in handle:
                    item=json.loads(line)
                    if item['kind']!='event':
                        continue
                    item['rank_log']=os.path.basename(path)
                    target.write(json.dumps(item)+'\n')
                    event_count+=1
                    if len(events)<100:
                        events.append(dict(rank_log=item['rank_log'],step=item['step'],
                            accepted_time=item['accepted_time'],
                            reasons=dict((key,dict(count=len(ids),first_16_ids=ids[:16]))
                                         for key,ids in item['values'].items())))
    if not rows:
        raise ValueError('no global accepted states')
    with open(os.path.join(output,'global.csv'),'w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    frame_count=0
    for path in glob.iglob(os.path.join(directory,'frames','*.npz')):
        # One frame is open at a time; complete histories are not loaded.
        with np.load(path,allow_pickle=False) as frame:
            if 'F' in frame:
                determinant=np.linalg.det(frame['F'])
                if not np.allclose(determinant,frame['J'],atol=1.e-12,rtol=1.e-10,equal_nan=True):
                    raise ValueError('F/J mismatch: '+path)
        frame_count+=1
    hdf=[]
    import h5py
    for path in glob.iglob(os.path.join(directory,'frames','*.h5')):
        if not h5py.is_hdf5(path):
            raise ValueError('invalid HDF5 signature: '+path)
        with h5py.File(path,'r') as handle:
            if 'w' not in handle or 'mesh' not in handle:
                raise ValueError('missing mixed state or mesh: '+path)
        hdf.append(os.path.basename(path))
    xdmf=[]
    for path in glob.iglob(os.path.join(os.path.dirname(directory),'mesh_output','*.xdmf')):
        ET.parse(path);xdmf.append(os.path.basename(path))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    t=[v['time'] for v in rows]
    axes[0,0].plot([v['volume'] for v in rows],[v['pressure'] for v in rows]);axes[0,0].set(xlabel='LV volume (source units)',ylabel='LV pressure (source units)')
    for key in ('J_min','J_max'):
        axes[0,1].plot(t,[v[key] for v in rows],label=key)
    for key in ('stretch_min','stretch_max','sigma_min'):
        axes[1,0].plot(t,[v[key] for v in rows],label=key)
    axes[1,1].plot([v['time'] for v in solves],[v['final_residual_l2'] for v in solves])
    axes[1,1].set_ylabel('Final assembled residual L2 (mixed units)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    axes[0,1].legend();axes[1,0].legend();fig.tight_layout()
    fig.savefig(os.path.join(output,'overview.png'),dpi=150);plt.close(fig)
    with open(os.path.join(directory,'completion.json')) as handle:
        completion=json.load(handle)
    result=dict(completion=completion,first_100_rank_event_rows=events,
        total_rank_event_rows=event_count,all_events_file='events.jsonl',solve_summaries=solves,
        frames_checked=frame_count,hdf_signatures_and_groups_checked=hdf,xdmf_parsed=xdmf,
        root_cause='unresolved',full_pipeline_validated=False,
        limitations=['HDF5 group checks are not independent DOLFIN FE reconstruction',
            'trial Newton telemetry and event causality are not established'])
    with open(os.path.join(output,'report.json'),'w') as handle:
        json.dump(result,handle,indent=2)
    with open(os.path.join(output,'Report.md'),'w') as handle:
        handle.write('# FE–MyoSim diagnostic output inspection\n\n')
        handle.write('Root cause: **unresolved**. Local output checks do not certify MPI/FE invariance.\n\n')
        handle.write('%d accepted states, %d frames, %d HDF5 files inspected.\n'%(len(rows),frame_count,len(hdf)))
        handle.write('See report.json, all rank logs, completion.json and overview.png.\n')
    if case['fe_cycle_diagnostic'].get('generate_event_causality',False):
        sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
        from analyze_fe_event_causality import analyze
        analyze(instruction,os.path.join(output,'event_causality'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--instruction',required=True)
    report(p.parse_args().instruction)
