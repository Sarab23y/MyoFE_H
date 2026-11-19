"""Freeze a new case from an authenticated baseline; never edit that baseline."""
from __future__ import print_function
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for part in iter(lambda: handle.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def prepare(baseline, template, output, mode, repo, image, coordinates=None, baseline_repo=None,interrupt_after_steps=None):
    with open(baseline) as handle:
        original = json.load(handle)
    with open(template) as handle:
        spec = json.load(handle)
    case = copy.deepcopy(original)
    if case['protocol']['time_step'] != [0.001]:
        raise ValueError('baseline timestep must be 0.001 s')
    solver_params=case['mesh'].get('solver',{}).get('params',{})
    if solver_params.get('debugging_mode',[False])[0]:
        raise ValueError('actual baseline selects manual Newton branch; native default-Newton observer must not silently replace it')
    if case['protocol']['perturbation']['perturbations']:
        raise ValueError('healthy baseline must not contain perturbations')
    if case['protocol']['baroreflex']['activations'] != [dict(t_start_s=[5], t_stop_s=[60])]:
        raise ValueError('expected authenticated 5-60 s baroreflex schedule')
    if case['protocol']['no_of_time_steps'] != [60000]:
        raise ValueError('expected 60-s baseline as starting point')
    for name in ('growth', 'fibrosis', 'fiber_reorientation'):
        if name in case['model']:
            raise ValueError('review remodeling module before using this preparer: '+name)
    case['output_handler'] = copy.deepcopy(spec['output_handler'])
    case['fe_cycle_diagnostic'] = copy.deepcopy(spec['fe_cycle_diagnostic'])
    output = os.path.realpath(output)
    if os.path.exists(output):
        raise ValueError('output already exists: '+output)
    case['fe_cycle_diagnostic']['directory'] = os.path.join(output, 'observer')
    case['fe_cycle_diagnostic']['enabled'] = mode not in ('control','pure-control')
    case['fe_cycle_diagnostic']['invariance_witness'] = mode == 'control'
    case['fe_cycle_diagnostic']['smoke_steps'] = 200 if mode != 'full' else 0
    if interrupt_after_steps is not None:
        if mode!='smoke' or not 1<=int(interrupt_after_steps)<200:
            raise ValueError('controlled interruption is only for a short enabled smoke, before step 200')
        case['fe_cycle_diagnostic']['diagnostic_abort_after_steps']=int(interrupt_after_steps)
    if mode != 'full':
        case['protocol']['no_of_time_steps'] = [199]
    if coordinates:
        with open(coordinates) as handle:
            points = json.load(handle)
        if len(points) != 4 or any(len(p) != 3 for p in points):
            raise ValueError('expected four authoritative reference coordinate triples')
        case['fe_cycle_diagnostic']['hotspot_coordinates'] = points
    elif mode == 'full':
        raise ValueError('full run requires verified historical hotspot coordinates')
    case['output_handler']['output_data_path'] = [os.path.join(output, 'sim_output', 'data.csv')]
    case['output_handler']['mesh_output_path'] = [os.path.join(output, 'mesh_output')]
    # Interpret the exact production mesh path relative to python_codes, just as production.
    raw = original['mesh']['mesh_path'][0]
    baseline_repo = baseline_repo or repo
    mesh = os.path.realpath(os.path.join(baseline_repo, 'python_codes', raw)) if original['mesh'].get('relative_path',[False])[0] else os.path.realpath(raw)
    if not os.path.isfile(mesh):
        raise ValueError('missing actual baseline mesh: '+mesh)
    with open(mesh,'rb') as handle:
        if handle.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('mesh is an LFS pointer, not the actual production HDF5 file: '+mesh)
    # Relocate only the path syntax; both cases now reference the same original
    # read-only mesh bytes regardless of the diagnostic worktree location.
    case['mesh']['relative_path'] = [False]
    case['mesh']['mesh_path'] = [mesh]
    if not os.path.isfile(image):
        raise ValueError('missing FEniCS image: '+image)
    # Conservative uncompressed envelope, not a filesystem compression promise.
    # 5008 points, <=128 states, five stage copies, 64 derived scalars, all 60001 steps.
    # Actual fixture shape/bytes must be measured on LCC before approving full run.
    frame = 5008*(5*(128+64)+64+9)*8 + 4*1024*1024
    full_steps=60001
    checkpoint_count=(full_steps-1)//int(case['fe_cycle_diagnostic']['checkpoint_stride'])+1
    routine_steps=set(range(0,full_steps,int(case['fe_cycle_diagnostic']['full_field_stride'])))
    checkpoint_steps=set(range(0,full_steps,int(case['fe_cycle_diagnostic']['checkpoint_stride'])))
    routine_count=len(routine_steps-checkpoint_steps)
    # Routine field NPZs also retain all stage arrays in the existing profile.
    # Counting only the derived fields here understated the old 49-GiB estimate.
    routine_payload=frame-4*1024*1024
    estimate = dict(assumed_points=5008, upper_population_columns=128,
        maximum_steps=60001, bytes_per_full_frame_envelope=frame,
        configured_steps=case['protocol']['no_of_time_steps'][0]+1,
        configured_persistent_trigger_worst_bytes=int(1.5*(case['protocol']['no_of_time_steps'][0]+1)*frame),
        bounded_rolling_frames=21, extra_array_memory_global_bytes=21*frame,
        persistent_trigger_worst_bytes=int(1.5*60001*frame),
        routine_full_field_frames=routine_count,checkpoint_frames=checkpoint_count,
        routine_field_payload_bytes=routine_payload,
        no_event_lower_planning_bytes=int(1.5*(checkpoint_count*frame+routine_count*routine_payload)),
        native_JSON_record_bytes_assumption=2048,
        native_JSON_max_iterations=solver_params.get('max_iter',[50])[0],
        native_JSON_60001_step_planning_bytes=int(1.5*60001*32*2*(solver_params.get('max_iter',[50])[0]+1)*2048),
        caveat='Partial file-payload planning, not a total disk guarantee. Excludes baseline/baroreflex CSV/JSON logs, postprocessing SQLite/CSV, backend memory and Python objects. Measure smoke sizes and scale; no runtime prediction without sacct.')
    os.makedirs(output)
    for name, value in [('instruction.json',case), ('storage_estimate.json',estimate),
                        ('baseline_input.json',original)]:
        with open(os.path.join(output,name),'w') as handle:
            json.dump(value,handle,indent=2)
    code = {}
    for source_root in ('python_codes','validation','postprocessing','tests'):
        for base, dirs, names in os.walk(os.path.join(repo,source_root)):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for name in names:
                if name.endswith(('.py','.sh','.slurm','.cpp')):
                    path = os.path.join(base,name)
                    code[os.path.relpath(path,repo)] = digest(path)
    provenance = dict(baseline_path=os.path.realpath(baseline),baseline_sha256=digest(baseline),
        input_sha256=digest(os.path.join(output,'instruction.json')),
        mesh_path=mesh,mesh_sha256=digest(mesh),image_path=image,image_sha256=digest(image),
        worktree=os.path.realpath(repo),source_files=code,
        git_head=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD']).decode().strip(),
        git_status=subprocess.check_output(['git','-C',repo,'status','--porcelain']).decode(),
        solver=original['mesh'].get('solver', {'source_defaults':'NSolver.default_solver_parameters'}),
        scientific_model_unchanged=case['model']==original['model'],
        scientific_mesh_unchanged_except_path=all(case['mesh'][key]==value for key,value
            in original['mesh'].items() if key not in ('mesh_path','relative_path')),
        baseline_worktree=os.path.realpath(baseline_repo),
        mode=mode, full_pipeline_validated=False)
    with open(os.path.join(output,'provenance.json'),'w') as handle:
        json.dump(provenance,handle,indent=2)
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-input','template','output','repo','image'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=['smoke','control','pure-control','full'],required=True)
    p.add_argument('--coordinates-json')
    p.add_argument('--baseline-repo',required=True,help='original worktree used to resolve the authenticated relative mesh path')
    p.add_argument('--interrupt-after-steps',type=int)
    args = p.parse_args()
    prepare(args.baseline_input,args.template,args.output,args.mode,args.repo,args.image,args.coordinates_json,args.baseline_repo,args.interrupt_after_steps)
    print('Prepared new case; NOT submitted and NOT certified for full execution: '+args.output)


if __name__ == '__main__':
    main()
