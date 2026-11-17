"""Stage-resolved event chronology, bounded frame memory and disk-backed joins.

Temporal precedence is reported as an observation, never a causal diagnosis.
Python >=3.7, NumPy, Matplotlib. Original outputs are opened read-only.
"""
import argparse
import csv
import glob
import json
import os
import re
import sqlite3
import numpy as np

STAGES=('before_advection','after_advection','before_kinetics',
        'after_kinetics_before_feedback','after_feedback')
ORDER=[('previous_accepted_FE',0),('before_advection',1),('after_advection',2),
       ('before_kinetics_consumed_calcium',3),('after_kinetics_before_feedback',4),
       ('after_feedback_assignment',5),('Newton_initial_and_trial_states',6),
       ('accepted_FE_before_hsl_old_reset',7),('hsl_old_reset_and_original_output',8),
       ('next_timestep_cellular_updates',9)]


def number(value):
    if value is None:return None
    value=float(value)
    return value if np.isfinite(value) else None


def relation(step,order,anchor):
    if step is None:return 'unobserved'
    if (step,order)<(anchor,7):return 'before accepted FE event'
    if (step,order)==(anchor,7):return 'at accepted FE event'
    return 'after accepted FE event'


def persistence(values,anchor):
    """Explicit descriptive rule: exact +50-ms sample; no missing-data filling."""
    by_step=dict(values)
    if anchor-1 not in by_step or anchor not in by_step or anchor+50 not in by_step:
        return dict(classification='unresolved: missing native before/onset/+50ms sample')
    jump=by_step[anchor]-by_step[anchor-1]
    if not np.isfinite(jump) or abs(jump)<=1.e-12:
        return dict(classification='unresolved: no finite nonzero jump')
    steps=list(range(anchor-1,anchor+51))
    if not all(step in by_step for step in steps):
        return dict(classification='unresolved: incomplete following 50ms')
    delta=np.diff([by_step[s] for s in steps])
    significant=delta[np.abs(delta)>=.1*abs(jump)]
    sign_changes=int(np.sum(np.sign(significant[1:])!=np.sign(significant[:-1])))
    retained=(by_step[anchor+50]-by_step[anchor-1])/jump
    classification=('oscillatory' if sign_changes>=2 else 'reversed' if retained<0 else
                    'persists' if retained>=.5 else 'partial recovery')
    return dict(classification=classification,retained_jump_fraction_at_50ms=number(retained),
                significant_increment_sign_changes=sign_changes)


def stage_value(frame,stage,key,index):
    name='stage__'+stage+'__'+key
    return frame[name][index] if name in frame else None


def population_change(left,right):
    if left is None or right is None:return None
    return number(np.sum(np.abs(np.asarray(right)-np.asarray(left))))


def observation(frame,index,previous,step):
    """One row has explicit source stages; vectors remain in original NPZ files."""
    def f(name):return number(frame[name][index]) if name in frame else None
    def s(stage,key):return number(stage_value(frame,stage,key,index))
    row=dict(step=step,legacy_time=number(frame['legacy_time']),accepted_time=number(frame['accepted_time']),
        previous_accepted_step=previous['step'] if previous else None,
        previous_native_consecutive=bool(previous and previous['step']==step-1),
        previous_FE_hsl=number(previous['hsl'][index]) if previous else None,
        previous_FE_stretch=number(previous['stretch'][index]) if previous else None,
        previous_FE_J=number(previous['J'][index]) if previous else None,
        cell_HSL_before_advection=s('before_advection','hs_length'),
        delta_HSL_supplied=s('before_advection','delta_hsl'),
        incoming_cb_feedback=s('before_advection','incoming_cb'),
        stored_cb_before_kinetics=s('before_kinetics','stored_cb'),
        stored_cb_after_feedback=s('after_feedback','stored_cb'),
        cell_HSL_after_advection=s('after_advection','hs_length'),
        calcium_consumed=s('before_kinetics','consumed_calcium'),
        calcium_cached_before_advection=s('before_advection','calcium'),
        overlap_before_advection=s('before_advection','raw_overlap'),
        overlap_after_advection=s('after_advection','raw_overlap'),
        overlap_before_kinetics=s('before_kinetics','raw_overlap'),
        overlap_after_kinetics=s('after_kinetics_before_feedback','stored_overlap'),
        population_L1_advection=population_change(stage_value(frame,'before_advection','y',index),
                                                 stage_value(frame,'after_advection','y',index)),
        population_L1_kinetics=population_change(stage_value(frame,'before_kinetics','y',index),
                                               stage_value(frame,'after_kinetics_before_feedback','y',index)),
        active_initial=f('active_newton_initial_PK2_coefficient'),
        active_equilibrium=f('active_equilibrium_PK2_coefficient'),
        accepted_hsl=f('hsl'),accepted_stretch=f('stretch'),accepted_J=f('J'),
        accepted_sigma_min=f('sigma_min'),Xi_PK2_ff=f('Xi_PK2_ff'),ECM_PK2_ff=f('ECM_PK2_ff'),
        active_output=f('active_output_PK2_coefficient'),active_stage_difference=f('active_stage_difference'),
        overlap_output=f('overlap'),overlap_cellular_HSL=f('overlap_cellular_hsl'),
        hsl_old_pre_reset=f('hsl_old_pre_reset'),hsl_old_post_reset=f('hsl_old_post_reset'))
    row['production_FE_HSL']=f('hsl_FE_production_list')
    row['active_initial_ff']=f('active_newton_initial_PK2_ff')
    row['active_equilibrium_ff']=f('active_equilibrium_PK2_ff')
    row['active_output_ff']=f('active_output_PK2_ff')
    row['production_FE_stretch']=f('stretch_FE_production_data')
    row['production_FE_HSL_minus_direct_F']=(row['production_FE_HSL']-row['accepted_hsl']
        if row['production_FE_HSL'] is not None and row['accepted_hsl'] is not None else None)
    row['production_FE_stretch_minus_direct_F']=(row['production_FE_stretch']-row['accepted_stretch']
        if row['production_FE_stretch'] is not None and row['accepted_stretch'] is not None else None)
    row['HSL_over_stretch']=(row['accepted_hsl']/row['accepted_stretch']
        if row['accepted_hsl'] is not None and row['accepted_stretch'] else None)
    raw=stage_value(frame,'before_kinetics','membrane_y',index)
    row['calcium_raw_before_kinetics']=number(raw[0]) if raw is not None and len(raw) else None
    for stage,label in [('before_advection','before'),('after_advection','after_advection'),
                        ('after_kinetics_before_feedback','after_kinetics')]:
        for key in ('myosin_sum','bound_sum','min_bin','negative_bins'):
            row[label+'_'+key]=s(stage,key)
    row['delta_stretch_native']=(row['accepted_stretch']-row['previous_FE_stretch']
        if row['previous_native_consecutive'] and row['accepted_stretch'] is not None and
        row['previous_FE_stretch'] is not None else None)
    row['delta_hsl_native']=(row['accepted_hsl']-row['previous_FE_hsl']
        if row['previous_native_consecutive'] and row['accepted_hsl'] is not None and
        row['previous_FE_hsl'] is not None else None)
    for i in range(3):
        for j in range(3):row['F%d%d'%(i,j)]=number(frame['F'][index,i,j])
    row['missing_stages']=';'.join(stage for stage in STAGES if 'stage__'+stage+'__y' not in frame)
    return row


def analyze(instruction,output,plots=True):
    with open(instruction) as handle:case=json.load(handle)
    config=case['fe_cycle_diagnostic'];directory=config['directory'];dt=case['protocol']['time_step'][0]
    if dt!=.001:raise ValueError('native 1-ms diagnostics required')
    before=int(round(config['pre_event_s']/dt));after=int(round(config['post_event_s']/dt))
    os.makedirs(output)
    db=sqlite3.connect(os.path.join(output,'event_index.sqlite'))
    db.executescript('''CREATE TABLE episode(id INTEGER PRIMARY KEY, rank INTEGER, gid INTEGER,
        start INTEGER, last INTEGER, lo INTEGER, hi INTEGER, reasons TEXT);
        CREATE INDEX episode_window ON episode(rank,lo,hi);
        CREATE TABLE native(rank INTEGER,step INTEGER,iteration INTEGER,payload TEXT,
            PRIMARY KEY(rank,step,iteration));
        CREATE TABLE global_context(step INTEGER PRIMARY KEY,payload TEXT);
        CREATE TABLE observations(episode INTEGER,step INTEGER,payload TEXT,PRIMARY KEY(episode,step));''')
    summary_file=None
    global_events=open(os.path.join(output,'global_newton_events.jsonl'),'w')
    global_event_count=0
    try:
        rank_paths=sorted(glob.glob(os.path.join(directory,'rank*.jsonl')))
        if not rank_paths:raise ValueError('no native rank logs')
        for path in rank_paths:
            rank=int(re.search(r'rank(\d+)',os.path.basename(path)).group(1));active={}
            with open(path) as handle:
                for line in handle:
                    item=json.loads(line);step=item.get('step')
                    if step is None:continue
                    if item['kind']=='newton_iteration':
                        value=item['values'];value['cardiac_cycle']=item.get('cardiac_cycle');db.execute('INSERT INTO native VALUES(?,?,?,?)',
                            (rank,step,value['iteration'],json.dumps(value)))
                    elif item['kind']=='global' and rank==0:
                        db.execute('INSERT INTO global_context VALUES(?,?)',(step,json.dumps(item['values'])))
                    elif item['kind']=='global_newton_event':
                        global_events.write(json.dumps(item)+'\n');global_event_count+=1
                    elif item['kind']=='event':
                        per_point={}
                        for reason,ids in item['values'].items():
                            for gid in ids:per_point.setdefault(int(gid),set()).add(reason)
                        for gid,reasons in per_point.items():
                            prior=active.get(gid)
                            if prior and step<=prior[1]+before+after:
                                eid,last,old=prior;reasons|=old
                                db.execute('UPDATE episode SET last=?,hi=?,reasons=? WHERE id=?',
                                    (step,step+after,json.dumps(sorted(reasons)),eid))
                            else:
                                cursor=db.execute('INSERT INTO episode(rank,gid,start,last,lo,hi,reasons) VALUES(?,?,?,?,?,?,?)',
                                    (rank,gid,step,step,max(0,step-before),step+after,json.dumps(sorted(reasons))))
                                eid=cursor.lastrowid
                            active[gid]=(eid,step,reasons)
            db.commit()
        with open(os.path.join(output,'native_newton.csv'),'w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['rank','step','iteration','native_record_json'])
            for row in db.execute('SELECT rank,step,iteration,payload FROM native ORDER BY rank,step,iteration'):writer.writerow(row)
        timeline=open(os.path.join(output,'event_timeline.csv'),'w',newline='');writer=None
        problems=[]
        try:
            for rank_path in rank_paths:
                rank=int(re.search(r'rank(\d+)',os.path.basename(rank_path)).group(1))
                with np.load(os.path.join(directory,'mapping_rank%04d.npz'%rank),allow_pickle=False) as m:
                    mapping={key:m[key].copy() for key in m.files}
                ids=mapping['global_ids'];lookup=dict((int(gid),i) for i,gid in enumerate(ids))
                if len(lookup)!=len(ids):raise ValueError('duplicate owned mapping IDs')
                paths={}
                for path in glob.glob(os.path.join(directory,'frames','*_rank%04d.npz'%rank)):
                    match=re.search(r'_(\d+)_rank',os.path.basename(path))
                    if not match:continue
                    step=int(match.group(1))
                    if step in paths:
                        with np.load(paths[step]) as a,np.load(path) as b:
                            for key in ('global_ids','accepted_time','F','J','stretch','hsl'):
                                if not np.array_equal(a[key],b[key]):raise ValueError('inconsistent duplicate frame '+path)
                    if step not in paths or os.path.basename(path).startswith('event_'):paths[step]=path
                previous=None
                for step,path in sorted(paths.items()):
                    with np.load(path,allow_pickle=False) as data:
                        frame={key:data[key].copy() for key in data.files}
                    if not np.array_equal(frame['global_ids'],ids):raise ValueError('frame changes point identities: '+path)
                    if abs(float(frame['accepted_time'])-float(frame['legacy_time'])-dt)>1.e-10:
                        raise ValueError('field recording offset: '+path)
                    if previous and abs(float(frame['accepted_time'])-previous['time']-(step-previous['step'])*dt)>1.e-8:
                        raise ValueError('inconsistent native timestep timestamp: '+path)
                    if not np.allclose(np.linalg.det(frame['F']),frame['J'],rtol=1.e-10,atol=1.e-12,equal_nan=True):
                        raise ValueError('saved F/J mismatch: '+path)
                    if 'fiber' in mapping:
                        reconstructed=np.linalg.norm(np.einsum('nij,nj->ni',frame['F'],mapping['fiber']),axis=1)
                        if not np.allclose(reconstructed,frame['stretch'],rtol=1.e-10,atol=1.e-12,equal_nan=True):
                            raise ValueError('saved F/fiber/stretch mismatch: '+path)
                    if 'hsl0' in mapping and not np.allclose(frame['hsl'],frame['stretch']*mapping['hsl0'],rtol=1.e-10,atol=1.e-8,equal_nan=True):
                        raise ValueError('saved HSL/stretch/reference mismatch: '+path)
                    episodes=list(db.execute('SELECT id,gid,start FROM episode WHERE rank=? AND lo<=? AND hi>=?',(rank,step,step)))
                    native=[json.loads(r[0]) for r in db.execute('SELECT payload FROM native WHERE rank=? AND step=? ORDER BY iteration',(rank,step))]
                    context=db.execute('SELECT payload FROM global_context WHERE step=?',(step,)).fetchone()
                    context=json.loads(context[0]) if context else {}
                    for eid,gid,start in episodes:
                        if gid not in lookup:raise ValueError('event ID absent from owned geometry')
                        i=lookup[gid];row=observation(frame,i,previous,step)
                        row.update(episode=eid,rank=rank,global_id=gid,
                            native_frame=path,
                            reference_x=number(mapping['coordinates'][i,0]),
                            reference_y=number(mapping['coordinates'][i,1]),
                            reference_z=number(mapping['coordinates'][i,2]),
                            global_cell=int(mapping['global_cells'][i]) if 'global_cells' in mapping else None,
                            newton_iterations=native[-1]['iteration'] if native else None,
                            newton_backend_converged=native[-1]['backend_converged'] if native else None,
                            native_telemetry_missing=not bool(native),
                            pressure=context.get('pressures',{}).get('pressure_ventricle'),
                            volume=context.get('volumes',{}).get('volume_ventricle'),
                            flow_out=context.get('flows',{}).get('flow_ventricle_to_aorta'),
                            flow_in=context.get('flows',{}).get('flow_veins_to_ventricle'),
                            checkpoint=os.path.join(directory,'frames','state_%08d.h5'%step))
                        if writer is None:
                            writer=csv.DictWriter(timeline,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row)
                        db.execute('INSERT INTO observations VALUES(?,?,?)',(eid,step,json.dumps(row,allow_nan=False)))
                    previous=dict(step=step,time=float(frame['accepted_time']),hsl=frame['hsl'],stretch=frame['stretch'],J=frame['J'])
                    db.commit()
        finally:timeline.close()
        summary_file=open(os.path.join(output,'event_summaries.json'),'w')
        summary_file.write('[');episode_count=0
        for eid,rank,gid,start,last,lo,hi,reasons in db.execute('SELECT * FROM episode ORDER BY rank,start,gid'):
            peak=db.execute('SELECT step,payload FROM observations WHERE episode=? ORDER BY step',(eid,))
            largest=None;count=0;missing_stages=0;missing_native=0;first_overlap=None;first_population=None
            for step,payload in peak:
                row=json.loads(payload);count+=1;missing_stages+=bool(row['missing_stages']);missing_native+=row['native_telemetry_missing']
                jump=row['delta_stretch_native']
                if jump is not None and (largest is None or abs(jump)>abs(largest[1])):largest=(step,jump,row['delta_hsl_native'])
                for key,order in [('overlap_before_advection',1),('overlap_after_advection',2),('overlap_before_kinetics',3),('overlap_after_kinetics',4),('overlap_output',8)]:
                    if row[key] is not None and row[key]<0 and first_overlap is None:first_overlap=(step,order,key)
                for key,order in [('before_negative_bins',1),('after_advection_negative_bins',2),('after_kinetics_negative_bins',4)]:
                    if row[key] is not None and row[key]>0 and first_population is None:first_population=(step,order,key)
            anchor=largest[0] if largest else start
            around=[json.loads(r[0]) for r in db.execute('SELECT payload FROM observations WHERE episode=? AND step>=? AND step<=? ORDER BY step',(eid,anchor-21,anchor+50))]
            vals=[(r['step'],r['accepted_stretch']) for r in around if r['accepted_stretch'] is not None]
            # Previous native state can supply the otherwise absent anchor-1 endpoint.
            for row in around:
                if row['previous_native_consecutive'] and row['previous_FE_stretch'] is not None:vals.append((row['step']-1,row['previous_FE_stretch']))
            summary=dict(episode=eid,rank=rank,global_id=gid,trigger_start_step=start,last_trigger_step=last,
                data_label=config.get('fixture_label','simulation records; historical provenance requires separate verification'),
                reasons=json.loads(reasons),requested_window=[lo,hi],native_rows=count,missing_native_steps=hi-lo+1-count,
                event_scope=('accepted_HSL_or_stretch' if set(json.loads(reasons)) & {'stretch_jump','low_stretch'}
                             else 'Newton_trial_or_other_diagnostic_event'),
                unavailable_startup_pre_steps=max(0,before-start),missing_stage_rows=missing_stages,missing_newton_rows=missing_native,
                largest_jump_interval=[anchor-1,anchor] if largest else None,
                largest_delta_stretch=largest[1] if largest else None,largest_delta_HSL_nm=largest[2] if largest else None,
                first_negative_overlap=first_overlap,first_negative_population=first_population,
                overlap_order=relation(first_overlap[0],first_overlap[1],anchor) if first_overlap else 'unobserved',
                population_order=relation(first_population[0],first_population[1],anchor) if first_population else 'unobserved',
                overlap_order_relative_to_trigger_onset=relation(first_overlap[0],first_overlap[1],start) if first_overlap else 'unobserved',
                population_order_relative_to_trigger_onset=relation(first_population[0],first_population[1],start) if first_population else 'unobserved',
                persistence=persistence(vals,anchor),classification='unresolved',
                causal_claim=False,independent_event_FE_reconstruction_verified=False)
            by_step=dict((row['step'],row) for row in around)
            onset=by_step.get(anchor);prior=by_step.get(anchor-1);following=by_step.get(anchor+1)
            def difference(row,key,reference,other_key=None):
                if row is None or reference is None:return None
                a,b=row.get(key),reference.get(other_key or key)
                return number(a-b) if a is not None and b is not None else None
            if onset:
                summary['point_identity']=dict(owner_rank=rank,global_id=gid,
                    reference_coordinates=[onset['reference_x'],onset['reference_y'],onset['reference_z']],
                    global_cell=onset['global_cell'])
                history=[json.loads(r[0]) for r in db.execute('SELECT payload FROM native WHERE rank=? AND step=? ORDER BY iteration',(rank,anchor))]
                summary['anchor_chronology']=dict(
                    previous_accepted_FE=dict(step=anchor-1,hsl=onset['previous_FE_hsl'],
                        J=onset['previous_FE_J'],stretch=onset['previous_FE_stretch'],
                        exact_native_previous=onset['previous_native_consecutive']),
                    before_FE=dict(calcium_consumed=onset['calcium_consumed'],
                        calcium_consumed_change_from_previous_step=difference(onset,'calcium_consumed',prior),
                        raw_membrane_calcium=onset['calcium_raw_before_kinetics'],
                        advection_delta_HSL=onset['delta_HSL_supplied'],
                        population_L1_advection=onset['population_L1_advection'],
                        population_L1_kinetics=onset['population_L1_kinetics'],
                        incoming_cb_feedback=onset['incoming_cb_feedback'],
                        stored_cb_before_kinetics=onset['stored_cb_before_kinetics'],
                        stored_cb_after_feedback=onset['stored_cb_after_feedback'],
                        active_at_initial_Newton_state=onset['active_initial']),
                    during_FE=dict(native_Newton_history=history,
                        active_equilibrium_minus_initial=difference(onset,'active_equilibrium',onset,'active_initial'),
                        point_stress_per_iteration='not recorded; only initial and converged coefficients'),
                    accepted_FE=dict(step=anchor,hsl=onset['accepted_hsl'],J=onset['accepted_J'],
                        stretch=onset['accepted_stretch'],active_equilibrium=onset['active_equilibrium'],
                        checkpoint=onset['checkpoint']),
                    after_FE=dict(active_output=onset['active_output'],
                        active_output_minus_equilibrium=onset['active_stage_difference'],
                        next_step_cell_HSL_after_advection=following['cell_HSL_after_advection'] if following else None,
                        next_step_overlap_before_kinetics=following['overlap_before_kinetics'] if following else None))
                changes=[]
                if onset['population_L1_advection'] is not None and onset['population_L1_advection']>1.e-12:
                    changes.append(dict(stage_order=2,quantity='population advection',normal_update_possible=True))
                if onset['calcium_consumed'] is not None and prior and prior['calcium_consumed'] is not None and abs(onset['calcium_consumed']-prior['calcium_consumed'])>1.e-15:
                    changes.append(dict(stage_order=3,quantity='consumed calcium',normal_update_possible=True))
                if onset['population_L1_kinetics'] is not None and onset['population_L1_kinetics']>1.e-12:
                    changes.append(dict(stage_order=4,quantity='population kinetics',normal_update_possible=True))
                changes.append(dict(stage_order=7,quantity='largest accepted stretch increment',normal_update_possible=True))
                summary['observed_changes_in_anchor_step_source_order']=sorted(changes,key=lambda x:x['stage_order'])
                summary['first_observed_change_is_not_a_cause']=True
            if episode_count:summary_file.write(',\n')
            summary_file.write(json.dumps(summary));episode_count+=1
            if plots and around:plot_episode(around,summary,output)
        summary_file.write(']\n');summary_file.close()
        jumps=0
        with open(os.path.join(output,'native_jump_intervals.csv'),'w',newline='') as handle:
            fields=('episode','rank','global_id','from_step','to_step','accepted_time','delta_stretch','delta_HSL_nm','accepted_J','backend_converged','following_50ms_json')
            jump_writer=csv.DictWriter(handle,fieldnames=fields);jump_writer.writeheader()
            threshold=config.get('thresholds',{}).get('delta_stretch',.02)
            for eid,step,payload in db.execute('SELECT episode,step,payload FROM observations ORDER BY episode,step'):
                row=json.loads(payload);delta=row['delta_stretch_native']
                if delta is None or abs(delta)<=threshold:continue
                following=[json.loads(r[0]) for r in db.execute('SELECT payload FROM observations WHERE episode=? AND step>=? AND step<=? ORDER BY step',(eid,step-1,step+50))]
                values=[(r['step'],r['accepted_stretch']) for r in following if r['accepted_stretch'] is not None]
                if row['previous_native_consecutive']:values.append((step-1,row['previous_FE_stretch']))
                jump_writer.writerow(dict(episode=eid,rank=row['rank'],global_id=row['global_id'],from_step=step-1,to_step=step,
                    accepted_time=row['accepted_time'],delta_stretch=delta,delta_HSL_nm=row['delta_hsl_native'],
                    accepted_J=row['accepted_J'],backend_converged=row['newton_backend_converged'],
                    following_50ms_json=json.dumps(persistence(values,step))))
                jumps+=1
        report=dict(episodes=episode_count,global_Newton_events=global_event_count,stage_order=ORDER,full_pipeline_validated=False,root_cause='unresolved',
            individual_native_jump_intervals=jumps,
            data_label=config.get('fixture_label','simulation records'),
            data_memory='one rank frame plus at most 72 native samples for one plot; joins/history on SQLite disk',
            normal_updates_are_not_abnormal_deviations=True,
            population_change_tolerance_for_reporting=1.e-12,
            limitations=['source-stage precedence does not demonstrate a causal mechanism',
                'trial J/stretch are global extrema; accepted point F/J are local',
                'event-specific independent checkpoint reconstruction remains required',
                'missing stages, rows and terminal post-context remain explicit',
                'exact valve events are not inferred from pressure/volume timestamps'])
        with open(os.path.join(output,'chronology_report.json'),'w') as handle:json.dump(report,handle,indent=2)
        with open(os.path.join(output,'Report.md'),'w') as handle:
            handle.write('# Source-stage event chronology\n\n%d episodes; root cause **unresolved**.\n\n'%episode_count)
            handle.write('event_timeline.csv contains native observations. native_newton.csv records trial histories.\n')
            handle.write('event_summaries.json reports exact largest intervals, missing context, overlap/population precedence and +50-ms behavior.\n')
            handle.write('Earlier normal kinetic/calcium changes do not establish the cause of later deformation.\n')
        return report
    finally:
        if summary_file is not None and not summary_file.closed:summary_file.close()
        global_events.close()
        db.close()


def plot_episode(rows,summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10));t=[r['accepted_time'] for r in rows]
    groups=[('accepted_hsl','cell_HSL_after_advection'),('accepted_J','accepted_stretch'),
        ('active_initial_ff','active_equilibrium_ff','active_output_ff','Xi_PK2_ff','ECM_PK2_ff'),
        ('calcium_consumed','calcium_raw_before_kinetics'),
        ('overlap_before_kinetics','after_kinetics_bound_sum','after_kinetics_min_bin'),
        ('newton_iterations',)]
    for ax,keys in zip(axes.flat,groups):
        for key in keys:ax.plot(t,[r.get(key) for r in rows],'.-',label=key,markersize=3)
        ax.legend(fontsize=7);ax.grid(alpha=.25);ax.set_xlabel('accepted FE time (s); cellular stages explicitly earlier')
    fig.suptitle('%s\nEpisode %d, rank %d, current global ID %d — native largest-jump window; causality unresolved'%
                 (summary['data_label'],summary['episode'],summary['rank'],summary['global_id']),fontsize=9)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(os.path.join(output,'event_%06d.png'%summary['episode']),dpi=150);plt.close(fig)
    history=summary.get('anchor_chronology',{}).get('during_FE',{}).get('native_Newton_history',[])
    if history:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        iterations=[r['iteration'] for r in history]
        axes[0].plot(iterations,[r['residual_l2'] for r in history],'.-');axes[0].set_ylabel('native residual L2 (mixed units)')
        for key in ('applied_displacement_l2','applied_displacement_max'):
            axes[1].plot(iterations,[(r.get('correction') or {}).get(key) for r in history],'.-',label=key)
        axes[1].set_ylabel('mesh-coordinate displacement units');axes[1].legend(fontsize=7)
        for key in ('J_min','J_max','stretch_min'):
            axes[2].plot(iterations,[(r.get('trial') or {}).get(key) for r in history],'.-',label='global trial '+key)
        accepted=summary['anchor_chronology']['accepted_FE']['J']
        axes[2].plot([iterations[-1]],[accepted],'r*',label='accepted local J');axes[2].legend(fontsize=7)
        for ax in axes:ax.set_xlabel('Newton residual evaluation iteration');ax.grid(alpha=.25)
        fig.suptitle(summary['data_label'],fontsize=8);fig.tight_layout()
        fig.savefig(os.path.join(output,'newton_%06d.png'%summary['episode']),dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--instruction',required=True);p.add_argument('--output',required=True)
    p.add_argument('--no-plots',action='store_true');args=p.parse_args();analyze(args.instruction,args.output,not args.no_plots)
