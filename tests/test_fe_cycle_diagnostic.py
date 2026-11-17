"""Synthetic contracts and source invariance, NOT a DOLFIN/MPI validation."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

REPO=Path(__file__).resolve().parents[1]
DEPENDENCIES=REPO/'python_codes/LV_simulation/dependencies'
sys.path.insert(0,str(DEPENDENCIES))
from fe_cycle_core import CycleLedger, EventWindow, deformation_metrics, population_metrics, triggers


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class CycleDiagnosticTests(unittest.TestCase):
    def test_actual_heart_rate_variable_cycles_initial_marker_is_not_completion(self):
        module=load('actual_hr',REPO/'python_codes/LV_simulation/heart_rate/heart_rate.py')
        hr=module.heart_rate(dict(t_active_period=[.003],t_quiescent_period=[.92],t_first_activation=[.1]))
        ledger=CycleLedger(30)
        beats=[]
        for step in range(60001):
            activation,beat,ed=hr.implement_time_step(.001)
            time=(step+1)*.001
            done=ledger.observe(time,beat)
            if beat:
                beats.append(time)
                if len(beats)==10:
                    hr.data['t_quiescent_period']=.71
                if len(beats)==1:
                    self.assertEqual(ledger.cycles,[])
            if done:
                break
        self.assertEqual(len(beats),31)
        self.assertEqual(len(ledger.cycles),30)
        self.assertEqual(ledger.cycles[-1]['end'],beats[-1])
        self.assertGreater(ledger.cycles[0]['end']-ledger.cycles[0]['start'],
                           ledger.cycles[-1]['end']-ledger.cycles[-1]['start'])

    def test_cycle_bad_time_and_target_rejected(self):
        with self.assertRaises(ValueError):CycleLedger(0)
        ledger=CycleLedger(2);ledger.observe(1.,1)
        with self.assertRaises(ValueError):ledger.observe(1.,0)

    def test_original_F_stretch_J_singular_values_and_no_mutation(self):
        F=np.asarray([np.eye(3),np.diag([.25,2.,2.]),np.diag([-1.,1.,1.])])
        original=F.copy();fiber=np.tile([1.,0.,0.],(3,1))
        v=deformation_metrics(F,fiber,[1000.,1000.,1000.])
        np.testing.assert_allclose(v['J'],[1,1,-1])
        np.testing.assert_allclose(v['stretch'],[1,.25,1])
        np.testing.assert_allclose(v['hsl'],[1000,250,1000])
        np.testing.assert_allclose(v['sigma_min'],[1,.25,1])
        np.testing.assert_allclose(v['condition'],[1,8,1])
        np.testing.assert_array_equal(F,original)

    def test_population_myosin_excludes_thin_sites(self):
        y=np.asarray([[.2,.3,.1,.4,.8,.2],[.2,.3,-.1,.6,.9,.1]])
        v=population_metrics(y,np.ones(2))
        np.testing.assert_allclose(v['myosin_sum'],1.)
        np.testing.assert_allclose(v['myosin_conservation_error'],0.)
        np.testing.assert_allclose(v['bound_sum'],.5)
        np.testing.assert_array_equal(v['negative_bins'],[0,1])

    def test_trigger_thresholds_observe_without_clipping(self):
        f=deformation_metrics(np.asarray([np.diag([.4,1,1]),np.diag([-1.,1,1])]),np.asarray([[1,0,0]]*2),[1000,1000])
        f['overlap']=np.asarray([-.1,1]);f['min_bin']=np.asarray([0,-.01])
        masks=triggers(f,np.asarray([.5,1]),dict(stretch=.5,J=.5,sigma_min=.1,delta_stretch=.02))
        self.assertEqual(masks['low_stretch'].tolist(),[True,False])
        self.assertEqual(masks['inverted'].tolist(),[False,True])
        self.assertEqual(masks['negative_overlap'].tolist(),[True,False])
        np.testing.assert_allclose(f['stretch'],[.4,1])

    def test_event_pre_post_union_no_duplicates_bounded_memory(self):
        window=EventWindow(20,50);written=[]
        for step in range(200):
            result=window.add(step,{'value':step},step in (40,60))
            written.extend(number for number,value in result)
            self.assertLessEqual(len(window.history),21)
        self.assertEqual(written,list(range(20,111)))
        self.assertEqual(len(written),len(set(written)))
        self.assertEqual(window.unfinished(199),0)

    def test_terminal_event_is_explicitly_incomplete(self):
        w=EventWindow(20,50)
        w.add(0,{},True)
        self.assertEqual(w.unfinished(0),50)

    def test_only_tagged_observation_blocks_touch_production(self):
        for name in ('python_codes/LV_simulation/LV_simulation.py',
                     'python_codes/LV_simulation/half_sarcomere/half_sarcomere.py'):
            original=subprocess.check_output(['git','-C',str(REPO),'show','HEAD:'+name]).decode()
            lines=[];inside=False
            for line in (REPO/name).read_text().splitlines(True):
                if line.strip()=='# FE_CYCLE_BEGIN':
                    self.assertFalse(inside);inside=True;continue
                if line.strip()=='# FE_CYCLE_END':
                    self.assertTrue(inside);inside=False;continue
                if not inside:lines.append(line)
            self.assertFalse(inside)
            self.assertEqual(''.join(lines).replace('\r\n','\n'),original.replace('\r\n','\n'))
        # Weak forms, solver, kinetics/advection, pressure, calcium and old launchers unchanged.
        tracked=subprocess.check_output(['git','-C',str(REPO),'diff','--name-only']).decode().splitlines()
        self.assertEqual(set(tracked),{
            'python_codes/LV_simulation/LV_simulation.py',
            'python_codes/LV_simulation/half_sarcomere/half_sarcomere.py'})

    def test_actual_half_sarcomere_method_preserves_order_and_results(self):
        path=REPO/'python_codes/LV_simulation/half_sarcomere/half_sarcomere.py'
        tree=ast.parse(path.read_text());cls=[n for n in tree.body if isinstance(n,ast.ClassDef)][0]
        method=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_simulation'][0]
        module=ast.Module(body=[method]);module.type_ignores=[]
        scope={};exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
        update=scope['update_simulation']
        class Myof:
            def __init__(self):
                self.y=np.asarray([.2,.3,.1,.4,.7,.3]);self.cb_stress=3.;self.pas_stress=4.
            def move_cb_distributions(self,delta):self.y[2]+=delta*.001
            def evolve_kinetics(self,dt,calcium):self.y[3]+=dt*calcium
        class Membrane:
            def __init__(self):self.data={'Ca_cytosol':1.}
            def implement_time_step(self,dt,activation):self.data['Ca_cytosol']+=dt*activation
        def fixture():return types.SimpleNamespace(myof=Myof(),memb=Membrane(),data={'hs_length':1000.})
        a,b=fixture(),fixture();seen=[]
        b._fe_observer=lambda stage,obj,detail=None:seen.append((stage,obj.myof.y.copy(),copy.deepcopy(detail)))
        aa=update(a,.001,-59.0309348,1,7.,8.)
        bb=update(b,.001,-59.0309348,1,7.,8.)
        np.testing.assert_array_equal(aa,bb);self.assertEqual(a.data,b.data)
        self.assertEqual(a.myof.total_stress,b.myof.total_stress)
        self.assertEqual([v[0] for v in seen],['before_advection','after_advection','before_kinetics','after_kinetics_before_feedback','after_feedback'])
        self.assertEqual(seen[2][2]['consumed_calcium'],1.001)
        self.assertEqual(seen[0][2]['incoming_cb'],7.)

    def test_templates_preserve_scientific_configuration_and_are_not_run_paths(self):
        base=json.loads((REPO/'demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json').read_text())
        for path in (REPO/'demos/ECM/sim_input').glob('*30cycles_FE_diagnostic*.json'):
            c=json.loads(path.read_text())
            for key in ('model','mesh','heart_rate'):
                self.assertEqual(c[key],base[key])
            for key in ('time_step','baroreflex','perturbation'):
                self.assertEqual(c['protocol'][key],base['protocol'][key])
            self.assertIn('REQUIRES_FE_CYCLE_PREPARATION',c['output_handler']['output_data_path'][0])

    def test_comparator_detects_row_offset_and_state_change(self):
        compare=load('smoke_compare',REPO/'validation/compare_fe_cycle_smoke.py')
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.csv';b=Path(tmp)/'b.csv'
            a.write_text('time,value\n0,1\n.001,2\n');b.write_text(a.read_text())
            self.assertEqual(compare.compare_csv(str(a),str(b))['rows'],2)
            b.write_text('time,value\n.001,1\n.002,2\n')
            with self.assertRaises(ValueError):compare.compare_csv(str(a),str(b))

    def test_full_runner_is_blocked_even_with_invented_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'case.json'
            p.write_text(json.dumps({'fe_cycle_diagnostic':dict(directory=tmp,smoke_steps=0)}))
            process=subprocess.run([sys.executable,str(REPO/'validation/run_fe_cycle_case.py'),str(p)],capture_output=True)
            self.assertNotEqual(process.returncode,0)
            self.assertIn(b'full run blocked',process.stderr)

    def test_preparer_preserves_baseline_and_refuses_existing_output(self):
        prepare=load('prepare_case',REPO/'validation/prepare_fe_cycle_case.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'image';image.write_bytes(b'synthetic image fixture, not executable')
            mesh=root/'mesh.h5';mesh.write_bytes(b'path/hash fixture only, not HDF5')
            case=json.loads((REPO/'demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json').read_text())
            case['mesh']['relative_path']=[False];case['mesh']['mesh_path']=[str(mesh)]
            baseline=root/'baseline.json';baseline.write_text(json.dumps(case));original=baseline.read_bytes()
            template=REPO/'demos/ECM/sim_input/base_instruction_Br_ECM_30cycles_FE_diagnostic.json'
            result=prepare.prepare(str(baseline),str(template),str(root/'new'),'smoke',str(REPO),str(image))
            self.assertEqual(baseline.read_bytes(),original)
            self.assertEqual(result['model'],case['model']);self.assertEqual(result['mesh'],case['mesh'])
            self.assertEqual(result['protocol']['no_of_time_steps'],[199])
            self.assertEqual(result['fe_cycle_diagnostic']['smoke_steps'],200)
            runner=load('prepared_runner',REPO/'validation/run_fe_cycle_case.py')
            self.assertEqual(runner.verify_prepared(str(root/'new/instruction.json'),result,str(REPO)),str(root/'new'))
            (root/'new/sim_output').mkdir()
            with self.assertRaises(ValueError):
                runner.verify_prepared(str(root/'new/instruction.json'),result,str(REPO))
            with self.assertRaises(ValueError):
                prepare.prepare(str(baseline),str(template),str(root/'new'),'smoke',str(REPO),str(image))

    def test_array_frame_timestamps_stages_and_exclusive_write(self):
        package=types.ModuleType('observer_test_package');package.__path__=[str(DEPENDENCIES)]
        with mock.patch.dict(sys.modules,{'observer_test_package':package}):
            runtime=load('observer_test_package.fe_cycle_diagnostic',DEPENDENCIES/'fe_cycle_diagnostic.py')
        class Comm:
            def allgather(self,x):return [x]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'frames').mkdir()
            obs=runtime.FECycleDiagnostic.__new__(runtime.FECycleDiagnostic)
            obs.root=tmp;obs.rank=0;obs.comm=Comm();obs.ids=np.asarray([42,91])
            frame=dict(fields={'J':np.asarray([1.,.8])},vectors={},
                       stages={'before_kinetics__consumed_calcium':np.asarray([.01,.02])},
                       label=.01,accepted_time=.011)
            obs.write_arrays(10,frame,'event')
            path=Path(tmp)/'frames/event_00000010_rank0000.npz'
            with np.load(str(path),allow_pickle=False) as data:
                self.assertEqual(float(data['legacy_time']),.01)
                self.assertEqual(float(data['accepted_time']),.011)
                np.testing.assert_array_equal(data['global_ids'],[42,91])
                np.testing.assert_allclose(data['stage__before_kinetics__consumed_calcium'],[.01,.02])
            with self.assertRaises(RuntimeError):obs.write_arrays(10,frame,'event')

    def test_read_only_checkpoint_roundtrip_api_contract_with_fake_backend(self):
        package=types.ModuleType('checkpoint_test_package');package.__path__=[str(DEPENDENCIES)]
        with mock.patch.dict(sys.modules,{'checkpoint_test_package':package}):
            runtime=load('checkpoint_test_package.fe_cycle_diagnostic',DEPENDENCIES/'fe_cycle_diagnostic.py')
        class Vector:
            def __init__(self):self.values=np.zeros(3)
            def get_local(self):return self.values.copy()
            def set_local(self,v):self.values=np.asarray(v).copy()
            def apply(self,unused):pass
        class Function:
            def __init__(self,space):self.v=Vector()
            def vector(self):return self.v
            def function_space(self):return 'synthetic_space'
        storage={};closed=[]
        class HDF:
            def __init__(self,comm,path,mode):self.path=path;self.mode=mode
            def write(self,obj,name):
                storage[(self.path,name)]=obj.vector().get_local() if isinstance(obj,Function) else 'mesh'
            def read(self,obj,name):obj.vector().set_local(storage[(self.path,name)])
            def close(self):closed.append(self.mode)
        class Comm:
            def allgather(self,x):return [x]
            def bcast(self,x,root=0):return x
        backend=types.ModuleType('dolfin');backend.Function=Function;backend.HDF5File=HDF
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules,{'dolfin':backend}):
            (Path(tmp)/'frames').mkdir()
            obs=runtime.FECycleDiagnostic.__new__(runtime.FECycleDiagnostic)
            obs.root=tmp;obs.rank=0;obs.comm=Comm();obs.ids=np.asarray([3,4,5])
            obs.mesh=types.SimpleNamespace(mpi_comm=lambda:None)
            obs.w=Function('space');obs.w.vector().set_local([7.,8.,9.])
            obs.functions={'w':obs.w};obs.checkpoint_frames=0;obs.event_frames=0;obs.emit=lambda *x:None
            frame=dict(fields={'J':np.ones(3)},vectors={'w':np.asarray([1.,2.,3.])},stages={},label=.1,accepted_time=.101)
            obs.write_frame(100,frame,True)
            np.testing.assert_array_equal(obs.w.vector().get_local(),[7,8,9])
            np.testing.assert_array_equal(storage[(str(Path(tmp)/'frames/state_00000100.h5'),'/w')],[1,2,3])
            self.assertEqual(closed,['w','r'])
            self.assertEqual(obs.checkpoint_frames,1)

    def test_postprocessor_synthetic_hdf_signature_and_nonroot_event(self):
        report=load('case_report',REPO/'postprocessing/report_fe_cycle_case.py')
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);observer=root/'observer';observer.mkdir();(observer/'frames').mkdir()
            global_values=dict(mechanics=dict(J_min=.8,J_max=1.1,stretch_min=.5,stretch_max=1.2,sigma_min=.4,inverted=0),
                pressures=dict(pressure_ventricle=80.),volumes=dict(volume_ventricle=100.),heart_rate=60.)
            line=dict(kind='global',accepted_time=.001,legacy_time=0.,values=global_values)
            (observer/'rank0000.jsonl').write_text(json.dumps(line)+'\n')
            event=dict(kind='event',accepted_time=.001,legacy_time=0.,step=0,values={'low_J':[91]})
            (observer/'rank0001.jsonl').write_text(json.dumps(event)+'\n')
            np.savez(str(observer/'frames/event_00000000_rank0000.npz'),F=np.asarray([np.diag([.8,1.,1.])]),J=np.asarray([.8]))
            with h5py.File(str(observer/'frames/state_00000000.h5'),'w') as handle:
                handle.create_group('w');handle.create_group('mesh')
                handle.attrs['fixture']='synthetic HDF signature only; not a DOLFIN checkpoint'
            (observer/'completion.json').write_text(json.dumps({'simulation_completed':False,'fixture':'synthetic'}))
            instruction=root/'input.json';instruction.write_text(json.dumps({'fe_cycle_diagnostic':{'directory':str(observer)}}))
            result=report.report(str(instruction))
            self.assertEqual(result['total_rank_event_rows'],1)
            self.assertEqual(result['first_100_rank_event_rows'][0]['rank_log'],'rank0001.jsonl')
            self.assertFalse(result['full_pipeline_validated'])
            self.assertEqual(result['root_cause'],'unresolved')
            self.assertTrue((root/'analysis/overview.png').is_file())


if __name__=='__main__':unittest.main()
