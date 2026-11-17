"""Dedicated legacy-FEniCS entry point. No automatic SLURM submission."""
from __future__ import print_function
import argparse
import json
import os
import sys
import hashlib


def verify_prepared(instruction, case, repo):
    root = os.path.dirname(os.path.dirname(case['output_handler']['output_data_path'][0]))
    with open(os.path.join(root,'provenance.json')) as handle:
        provenance = json.load(handle)
    with open(instruction,'rb') as handle:
        if hashlib.sha256(handle.read()).hexdigest()!=provenance['input_sha256']:
            raise ValueError('prepared instruction changed since archival')
    if os.path.realpath(repo)!=os.path.realpath(provenance['worktree']):
        raise ValueError('case was prepared for a different worktree')
    for relative, expected in provenance['source_files'].items():
        with open(os.path.join(repo,relative),'rb') as handle:
            if hashlib.sha256(handle.read()).hexdigest()!=expected:
                raise ValueError('source changed since preparation: '+relative)
    for name in ('sim_output','mesh_output','observer','witness','final_mixed.h5','run_started.json'):
        if os.path.exists(os.path.join(root,name)):
            raise ValueError('refusing to reuse output: '+os.path.join(root,name))
    return root


def capture_initial_state(simulation,directory):
    """Common read-only initialization witness, not an FE/restart checkpoint."""
    import numpy as np
    from LV_simulation.dependencies.fe_cycle_diagnostic import clean_json
    payload={}
    def visit(path,item):
        if isinstance(item,dict):
            for key,value in item.items():visit(path+'__'+str(key),value)
        elif isinstance(item,(list,tuple)):
            for index,value in enumerate(item):visit(path+'__'+str(index),value)
        elif hasattr(item,'vector') and hasattr(item,'function_space'):
            payload['function'+path]=item.vector().get_local().copy()
            payload['element'+path]=np.asarray(str(item.function_space().ufl_element()))
    error=None
    try:
        visit('',simulation.mesh.model['functions'])
        objects=simulation.hs_objs_list
        if objects:
            for label,attribute in (('hs','data'),('myof','data'),('membrane','data'),
                                    ('myof_implementation','implementation'),('membrane_implementation','implementation')):
                members=[(obj if label=='hs' else obj.myof if label.startswith('myof') else obj.memb) for obj in objects]
                dictionaries=[getattr(member,attribute) for member in members]
                for key,value in dictionaries[0].items():
                    if np.isscalar(value) and not isinstance(value,(str,bytes)):
                        payload[label+'__'+str(key)]=np.asarray([item[key] for item in dictionaries])
        payload.update(global_ids=np.asarray(simulation.dofmap),
            cellular_y=np.asarray([h.myof.y for h in simulation.hs_objs_list]),
            membrane_y=np.asarray([h.memb.y for h in simulation.hs_objs_list]),
            cellular_hsl=np.asarray([h.data['hs_length'] for h in simulation.hs_objs_list]),
            physiology_json=np.asarray(json.dumps(clean_json(dict(circulation=simulation.circ.data,
                heart_rate=simulation.hr.data,baroreflex=simulation.br.data if simulation.br else None)),sort_keys=True,allow_nan=False)))
        target=os.path.join(directory,'initial_rank%04d.npz'%simulation.comm.Get_rank())
        descriptor=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        with os.fdopen(descriptor,'wb') as handle:np.savez_compressed(handle,**payload)
    except Exception as exc:error=str(exc)
    errors=simulation.comm.allgather(error)
    if any(errors):raise RuntimeError('initial-state evidence write failed: '+str(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instruction')
    parser.add_argument('--readiness-report')
    args = parser.parse_args()
    with open(args.instruction) as handle:
        case = json.load(handle)
    config = case['fe_cycle_diagnostic']
    if config['directory'] == 'CONFIGURE_AT_PREPARATION':
        parser.error('use prepare_fe_cycle_case.py first')
    if not config['smoke_steps']:
        # Requirements are deliberately not manufactured from unit-test results.
        required = ('same_source_and_input', 'mapping_1_and_32', 'cg2_roundtrip_1_and_32',
                    'invariance_1_and_32', 'newton_trial_and_correction_observer',
                    'output_finalization', 'historical_hotspots_verified', 'storage_checked')
        if not args.readiness_report:
            parser.error('full run blocked pending actual LCC validation and user approval')
        with open(args.readiness_report) as handle:
            readiness = json.load(handle)
        if any(readiness.get(key) is not True for key in required):
            parser.error('full run readiness requirements are not satisfied')
        parser.error('full run blocked: actual container telemetry/reconstruction/invariance evidence and explicit submission approval are required')
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = verify_prepared(args.instruction, case, repo)
    sys.path.insert(0, os.path.join(repo,'python_codes'))
    from dolfin import parameters
    from mpi4py import MPI
    from LV_simulation.LV_simulation import LV_simulation
    from LV_simulation.dependencies.recode_dictionary import recode
    MPI.COMM_WORLD.Barrier()
    error = None
    if MPI.COMM_WORLD.Get_rank()==0:
        try:
            descriptor=os.open(os.path.join(result,'run_started.json'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
            with os.fdopen(descriptor,'w') as handle:
                plugin=os.environ.get('MYOFE_NEWTON_OBSERVER')
                plugin_digest=None
                if plugin:
                    with open(plugin,'rb') as binary:plugin_digest=hashlib.sha256(binary.read()).hexdigest()
                json.dump(dict(ranks=MPI.COMM_WORLD.Get_size(),instruction=args.instruction,
                    native_hook=plugin,native_hook_sha256=plugin_digest,
                    measurement_witness=bool(config.get('invariance_witness')),
                    pure_disabled=not bool(config['enabled'] or config.get('invariance_witness'))),handle)
        except Exception as exc:
            error=str(exc)
    error=MPI.COMM_WORLD.bcast(error,root=0)
    if error:
        raise RuntimeError(error)
    simulation = None
    error = None
    success = False
    try:
        native = recode(case)
        simulation = LV_simulation(MPI.COMM_WORLD,native)
        capture_initial_state(simulation,result)
        simulation.run_simulation(native['protocol'],native['output_handler'])
        # Identical post-run-only capture for enabled/disabled invariance checks.
        # It is outside the physical timestep loop in both cases.
        import numpy as np
        from dolfin import HDF5File
        result = os.path.dirname(os.path.dirname(case['output_handler']['output_data_path'][0]))
        f = simulation.mesh.model['functions']
        target = os.path.join(result,'final_mixed.h5')
        exists = MPI.COMM_WORLD.bcast(os.path.exists(target) if MPI.COMM_WORLD.Get_rank()==0 else None,root=0)
        if exists:
            raise RuntimeError('refusing to replace final mixed checkpoint')
        writer = HDF5File(simulation.mesh.model['mesh'].mpi_comm(),str(target),'w')
        try:
            writer.write(simulation.mesh.model['mesh'],'/mesh')
            writer.write(f['w'],'/w')
        finally:
            writer.close()
        np.savez(os.path.join(result,'final_rank%04d.npz'%MPI.COMM_WORLD.Get_rank()),
            legacy_time=simulation.data['time']-.001, accepted_time=simulation.data['time'],
            global_ids=simulation.dofmap, mixed_owned=f['w'].vector().get_local(),
            mixed_range=f['w'].vector().local_range(), hsl=simulation.hs_length_list,
            stretch=simulation.data['fiber_stretch_raw'], y=simulation.y_vec,
            calcium=np.asarray([h.memb.data['Ca_cytosol'] for h in simulation.hs_objs_list]),
            cb_feedback=simulation.cb_stress_list, passive_feedback=simulation.pass_stress_list)
        success = True
    except BaseException as exc:
        error = '%s: %s' % (type(exc).__name__,exc)
        raise
    finally:
        if simulation is not None:
            observer = getattr(simulation,'_fe_cycle',None)
            if observer is not None:
                observer.finish(success,error)
            if not success:
                for name in ('solution_mesh','growth_mesh'):
                    handle = getattr(simulation,name,None)
                    if handle is not None:
                        handle.close()


if __name__ == '__main__':
    main()
