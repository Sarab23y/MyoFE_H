"""Read-only output integrity audit. File existence alone never proves FE recovery."""
import argparse
import csv
import glob
import hashlib
import json
import os
import xml.etree.ElementTree as ET
import h5py
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for data in iter(lambda:handle.read(1048576),b''):h.update(data)
    return h.hexdigest()


def check(instruction,interrupted=False):
    with open(instruction) as handle:case=json.load(handle)
    root=os.path.dirname(os.path.dirname(case['output_handler']['output_data_path'][0]))
    with open(os.path.join(root,'run_started.json')) as handle:started=json.load(handle)
    ranks=started['ranks'];config=case['fe_cycle_diagnostic']
    observer=config['directory'] if config['enabled'] else os.path.join(root,'witness')
    completion=None
    if config['enabled']:
        with open(os.path.join(observer,'completion.json')) as handle:completion=json.load(handle)
    if interrupted:
        if not completion or completion['last_step']+1!=config.get('diagnostic_abort_after_steps'):
            raise ValueError('controlled interruption did not reach its configured accepted boundary')
        if 'intentional diagnostic smoke interruption' not in (completion.get('error') or ''):
            raise ValueError('unrelated failure must not pass as a controlled interruption')
    with open(os.path.join(root,'sim_output/output_metadata.json')) as handle:metadata=json.load(handle)
    if bool(metadata.get('complete'))==interrupted:
        raise ValueError('buffered completion metadata disagrees with expected completion/interruption')
    steps=(completion['last_step']+1 if interrupted and completion else
           config['smoke_steps'] or (completion['last_step']+1 if completion else 0))
    if steps<=0:raise ValueError('no completed timestep convention')
    csvs={}
    static_names={'lx_data.csv','ly_data.csv','lz_data.csv'}
    id_path=os.path.join(root,'sim_output/quadrature_global_dof.npy')
    expected_ids=[str(int(v)) for v in np.load(id_path)] if os.path.isfile(id_path) else None
    for field in case['output_handler'].get('spatial_csv_fields',[]):
        if not os.path.isfile(os.path.join(root,'sim_output',field+'_data.csv')):
            raise ValueError('configured spatial output file missing: '+field)
    for path in glob.glob(os.path.join(root,'sim_output','*.csv')):
        stride=case['output_handler']['global_csv_frequency_n'][0] if os.path.basename(path)=='data.csv' else case['output_handler']['spatial_csv_frequency_n'][0]
        count=0;previous=None
        with open(path,newline='') as handle:
            reader=csv.DictReader(handle)
            if not reader.fieldnames or reader.fieldnames.count('time')!=1:raise ValueError('missing/duplicate time header')
            if os.path.basename(path)!='data.csv' and expected_ids is not None:
                actual_ids=[value for value in reader.fieldnames if value not in ('','time','Unnamed: 0')]
                if actual_ids!=expected_ids:raise ValueError('spatial CSV header point identity/order mismatch: '+path)
            for row in reader:
                value=float(row['time'])
                if not np.isfinite(value) or abs(value-count*stride*.001)>1.e-8:
                    raise ValueError('missing/duplicate/off-grid timestamp in '+path)
                previous=value;count+=1
        expected=1 if os.path.basename(path) in static_names else (steps-1)//stride+1
        if count!=expected:raise ValueError('wrong row count in '+path)
        csvs[os.path.basename(path)]=dict(rows=count,last_legacy_time=previous,
            kind='reference_geometry' if os.path.basename(path) in static_names else 'time_history')
    if 'data.csv' not in csvs:raise ValueError('no global CSV')
    if metadata.get('global_rows_written')!=steps:raise ValueError('global final batch not flushed')
    expected_spatial=(steps-1)//case['output_handler']['spatial_csv_frequency_n'][0]+1
    if metadata.get('spatial_rows_written')!=expected_spatial:raise ValueError('spatial final batch not flushed')
    hdfs=[]
    for base,dirs,files in os.walk(root):
        for name in files:
            if not name.endswith(('.h5','.hdf5')):continue
            path=os.path.join(base,name)
            if not h5py.is_hdf5(path):raise ValueError('invalid HDF5 signature: '+path)
            with h5py.File(path,'r') as handle:
                list(handle.keys())
            hdfs.append(os.path.relpath(path,root))
    if not hdfs:raise ValueError('no HDF5 files')
    xmls=[]
    for path in glob.glob(os.path.join(root,'mesh_output','*.xdmf')):
        tree=ET.parse(path)
        for item in tree.iter('DataItem'):
            if item.attrib.get('Format')!='HDF':continue
            text=(item.text or '').strip()
            if ':' not in text:raise ValueError('invalid XDMF HDF reference')
            filename,dataset=text.split(':',1)
            filename=os.path.join(os.path.dirname(path),filename)
            with h5py.File(filename,'r') as handle:
                if dataset not in handle:raise ValueError('XDMF references missing dataset: '+text)
        xmls.append(os.path.basename(path))
    if case['output_handler'].get('mesh_object_to_save') and not xmls:
        raise ValueError('expected closed XDMF document is absent')
    reconstruction={}
    if config['enabled'] and not interrupted:
        for n in ([1,32] if ranks==32 else [1]):
            path=os.path.join(observer,'reopen_%d.json'%n)
            with open(path) as handle:result=json.load(handle)
            if result.get('passed') is not True or result.get('ranks')!=n:
                raise ValueError('checkpoint FE reconstruction not demonstrated at rank count '+str(n))
            if result.get('state_sha256')!=digest(result['state_path']):
                raise ValueError('reconstruction report is stale/unbound to checkpoint bytes')
            reconstruction[str(n)]=result
    if completion:
        cycles=completion['cycles']
        if len(cycles)!=completion['completed_cycles']:raise ValueError('cycle manifest count mismatch')
        for i,cycle in enumerate(cycles):
            if cycle['cycle']!=i+1 or cycle['end']<=cycle['start']:raise ValueError('invalid complete-cycle interval')
            if i and abs(cycle['start']-cycles[i-1]['end'])>1.e-10:raise ValueError('discontinuous cycle intervals')
        if not config['smoke_steps'] and (len(cycles)!=30 or not completion['simulation_completed']):
            raise ValueError('full case has not completed exactly 30 intervals')
    frame_count=0;event_gaps=[];all_triggers=set()
    if config['enabled']:
        for rank in range(ranks):
            logpath=os.path.join(observer,'rank%04d.jsonl'%rank)
            trigger_steps=[];statuses=0
            with open(logpath) as handle:
                for line in handle:
                    item=json.loads(line)
                    if item['kind']=='event':trigger_steps.append(item['step'])
                    if item['kind']=='newton_solve_status' and item['values']['status']=='converged':statuses+=1
            if statuses!=steps:raise ValueError('missing successful per-step native Newton status')
            # A trigger on another rank causes the same global event-frame union.
            # Check every rank using the global union, not just local trigger IDs.
            frame_count+=len(glob.glob(os.path.join(observer,'frames','event_*_rank%04d.npz'%rank)))
        for path in glob.glob(os.path.join(observer,'rank*.jsonl')):
            with open(path) as handle:
                for line in handle:
                    item=json.loads(line)
                    if item['kind'] in ('event','global_newton_event'):all_triggers.add(item['step'])
        for step in sorted(all_triggers):
            lo=max(0,step-int(round(config['pre_event_s']/.001)))
            hi=min(steps-1,step+int(round(config['post_event_s']/.001)))
            for number in range(lo,hi+1):
                for rank in range(ranks):
                    path=os.path.join(observer,'frames','event_%08d_rank%04d.npz'%(number,rank))
                    if not os.path.isfile(path):event_gaps.append(dict(trigger=step,missing_step=number,rank=rank))
        if event_gaps:raise ValueError('event window missing frames: '+str(event_gaps[:10]))
    total=sum(os.path.getsize(os.path.join(base,name)) for base,dirs,files in os.walk(root) for name in files)
    startup_missing=max([0]+[max(0,int(round(config['pre_event_s']/.001))-s) for s in all_triggers])
    terminal_missing=completion.get('missing_post_event_steps',0) if completion else 0
    return dict(output_integrity_checks_passed=True,expected_interruption=interrupted,
        csv=csvs,hdf_signatures_readable=hdfs,xdmf_references_verified=xmls,
        independent_FE_reconstruction=reconstruction,cycles=completion['completed_cycles'] if completion else None,
        event_rank_frames=frame_count,terminal_post_context_missing=terminal_missing,
        startup_pre_context_missing=startup_missing,
        event_context_complete=not bool(terminal_missing or startup_missing),
        actual_output_bytes=total,full_pipeline_validated=False,
        caveats=['signature/XML checks are separate from the supplied independent FE reconstruction reports',
                 'manufactured cycle tests do not certify actual 30-cycle driver termination'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--instruction',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expect-interrupted',action='store_true');args=p.parse_args()
    result=check(args.instruction,args.expect_interrupted)
    descriptor=os.open(args.output,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    with os.fdopen(descriptor,'w') as handle:json.dump(result,handle,indent=2)
