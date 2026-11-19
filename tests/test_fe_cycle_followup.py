"""Protocol/chronology fixtures are synthetic, not FE or historical validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import ast
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


telemetry=load('native_telemetry_test',ROOT/'python_codes/LV_simulation/dependencies/native_newton_telemetry.py')
causal=load('causal_test',ROOT/'postprocessing/analyze_fe_event_causality.py')


def make_stage_fixture(root):
    """A known shortening at step30; negative overlap follows advection at31."""
    root=Path(root);observer=root/'observer';observer.mkdir();(observer/'frames').mkdir()
    config=dict(directory=str(observer),pre_event_s=.02,post_event_s=.05,
        fixture_label='SYNTHETIC stage-schema fixture: not a simulation or historical populations')
    instruction=root/'input.json';instruction.write_text(json.dumps(dict(fe_cycle_diagnostic=config,protocol={'time_step':[.001]})))
    for rank,gid in ((0,5),(1,91)):
        np.savez(str(observer/('mapping_rank%04d.npz'%rank)),global_ids=np.asarray([gid]),
            coordinates=np.asarray([[rank,0.,0.]]),fiber=np.asarray([[1.,0,0]]),hsl0=np.asarray([1000.]))
        with (observer/('rank%04d.jsonl'%rank)).open('w') as log:
            for step in range(90):
                def emit(kind,values):log.write(json.dumps(dict(kind=kind,step=step,legacy_time=step*.001,accepted_time=(step+1)*.001,values=values))+'\n')
                def lam(s):return .4 if rank==1 and s>=30 else 1.
                current,previous,old=lam(step),lam(step-1),lam(step-2)
                if rank==1 and step==30:emit('event',{'stretch_jump':[gid]})
                if rank==0:emit('global',dict(mechanics={},pressures={'pressure_ventricle':80+step},volumes={'volume_ventricle':100-step*.1},flows={'flow_ventricle_to_aorta':1.,'flow_veins_to_ventricle':0.}))
                for iteration in range(2):emit('newton_iteration',dict(iteration=iteration,residual_l2=10. if iteration==0 else 1.e-9,
                    relative_residual=1. if iteration==0 else 1.e-10,backend_converged=iteration==1,
                    cumulative_linear_iterations=iteration,correction=None if iteration==0 else dict(applied_displacement_l2=abs(current-previous),applied_displacement_max=abs(current-previous)),
                    trial=dict(J_min=previous if iteration==0 else current,J_max=1.,stretch_min=current)))
                y=np.asarray([.2,.3,.1,.4,.8,.2])
                adv=y.copy();adv[1]+=1.e-4;adv[2]-=1.e-4
                kin=adv.copy();kin[1]-=1.e-5;kin[3]+=1.e-5
                F=np.asarray([np.diag([current,1.,1.])])
                frame=dict(global_ids=np.asarray([gid]),legacy_time=step*.001,accepted_time=(step+1)*.001,
                    F=F,J=np.asarray([current]),stretch=np.asarray([current]),hsl=np.asarray([1000*current]),
                    sigma_min=np.asarray([current]),active_newton_initial_PK2_coefficient=np.asarray([10.]),
                    active_equilibrium_PK2_coefficient=np.asarray([8. if current<1 else 10.]),
                    active_output_PK2_coefficient=np.asarray([10.]),active_stage_difference=np.asarray([2. if current<1 else 0.]),
                    active_newton_initial_PK2_ff=np.asarray([10.]),
                    active_equilibrium_PK2_ff=np.asarray([8. if current<1 else 10.]),active_output_PK2_ff=np.asarray([10.]),
                    hsl_FE_production_list=np.asarray([1000*current]),stretch_FE_production_data=np.asarray([current]),
                    Xi_PK2_ff=np.asarray([0.]),ECM_PK2_ff=np.asarray([100.]),overlap=np.asarray([-.1 if previous<.5 else 1.]),
                    overlap_cellular_hsl=np.asarray([1000*previous]),hsl_old_pre_reset=np.asarray([1000*previous]),hsl_old_post_reset=np.asarray([1000*current]))
                for stage,vector in zip(causal.STAGES,[y,adv,adv,kin,kin]):
                    length=old if stage=='before_advection' else previous
                    values=dict(y=vector,hs_length=1000*length,delta_hsl=1000*(previous-old),incoming_cb=10.,stored_cb=10.,
                        raw_overlap=-.1 if length<.5 else 1.,stored_overlap=-.1 if length<.5 else 1.,
                        consumed_calcium=step*1.e-7,calcium=(step-1)*1.e-7,membrane_y=np.asarray([step*2.e-7,1.e-3]),
                        myosin_sum=np.sum(vector[:-2]),bound_sum=np.sum(vector[2:-2]),min_bin=np.min(vector),negative_bins=int(np.sum(vector<0)))
                    for key,value in values.items():frame['stage__'+stage+'__'+key]=np.asarray([value])
                np.savez_compressed(str(observer/'frames'/('event_%08d_rank%04d.npz'%(step,rank))),**frame)
    return instruction


class NativeTelemetryTests(unittest.TestCase):
    def monitor(self):
        class Comm:
            def allgather(self,x):return [x]
        m=telemetry.NewtonTelemetry.__new__(telemetry.NewtonTelemetry)
        self.u=np.asarray([0.,0.]);self.events=[]
        m.comm=Comm();m.displacement=lambda:self.u;m.emit=lambda *x:self.events.append(x)
        m.trial_metrics=lambda:dict(J_min=.4,J_max=1.1,stretch_min=.4)
        m.start();return m

    def test_native_sequence_uses_displacement_correction_and_original_relaxation(self):
        m=self.monitor();m._safe_callback(0,0,10.,1.,float('nan'),0,0)
        m._safe_callback(1,0,0.,0.,.5,0,1);self.u=np.asarray([-3.,-4.])
        m._safe_callback(2,1,0.,0.,.5,0,1);m._safe_callback(0,1,1.e-9,1.e-10,float('nan'),1,1)
        result=m.finish(True)
        self.assertEqual(result['iterations'],1)
        self.assertEqual(result['records'][1]['correction']['applied_displacement_l2'],5.)
        self.assertEqual(result['records'][1]['correction']['linear_displacement_l2'],10.)
        self.assertEqual(result['records'][1]['correction']['applied_displacement_max'],4.)
        self.assertEqual(result['records'][1]['trial']['J_min'],.4)
        self.assertEqual(result['records'][1]['state_category'],'trial_residual_evaluation')

    def test_missing_interposition_and_dropped_iteration_fail_closed(self):
        m=self.monitor()
        with self.assertRaises(RuntimeError):m.finish(True)
        m=self.monitor();m._safe_callback(0,1,1.,1.,1.,0,0)
        self.assertIsNotNone(m.error)
        with self.assertRaises(RuntimeError):m.finish(True)
        m=self.monitor();m._safe_callback(0,0,10.,1.,1.,0,0)
        m._safe_callback(0,1,0.,0.,1.,1,1)
        self.assertIn('displacement-update',m.error)
        with self.assertRaises(RuntimeError):m.finish(True)

    def test_callback_exception_is_deferred_not_unwound_through_solver(self):
        m=self.monitor()
        def fail(*x):raise IOError('fixture disk full')
        m.emit=fail;m._safe_callback(0,0,1.,1.,1.,0,0)
        self.assertIn('disk full',m.error)
        with self.assertRaises(RuntimeError):m.finish(True)

    def test_zero_iteration_convergence_and_failure_status(self):
        m=self.monitor();m._safe_callback(0,0,0.,float('nan'),1.,1,0)
        self.assertEqual(m.finish(True)['iterations'],0)
        m=self.monitor();m._safe_callback(0,0,1.,1.,1.,0,0)
        self.assertEqual(m.finish(False)['status'],'solver_failed')
        m=self.monitor();m._safe_callback(0,0,1.,1.,1.,0,0)
        self.assertEqual(m.finish(True)['status'],'backend_returned_nonconverged')
        m=self.monitor();m._safe_callback(9,0,0.,0.,0.,0,0)
        with self.assertRaises(RuntimeError):m.finish(True)

    def test_displacement_norm_global_reduction_excludes_pressure(self):
        m=self.monitor()
        class TwoRanks:
            def allgather(self,x):return [x,(144.,12.)] if isinstance(x,tuple) else [x,None]
        m.comm=TwoRanks();m.observe(1,0,0.,0.,1.,0,0);self.u=np.asarray([3.,4.]);m.observe(2,1,0.,0.,1.,0,1)
        self.assertEqual(m.correction['applied_displacement_l2'],13.)
        self.assertEqual(m.correction['applied_displacement_max'],12.)

    def test_native_source_delegates_each_original_method_once(self):
        source=(ROOT/'validation/native_newton_observer.cpp').read_text()
        self.assertEqual(source.count('forward(this,'),2)
        self.assertIn('RTLD_NEXT',source)
        self.assertNotIn('x.axpy',source);self.assertNotIn('assemble(',source)
        self.assertNotIn('parameters["convergence_criterion"] =',source)


class EventChronologyTests(unittest.TestCase):
    def test_source_stages_overlap_follows_shortening_and_exact_50ms_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            instruction=make_stage_fixture(tmp)
            result=causal.analyze(str(instruction),str(Path(tmp)/'analysis'),plots=False)
            self.assertEqual(result['episodes'],1)
            summary=json.loads((Path(tmp)/'analysis/event_summaries.json').read_text())[0]
            self.assertEqual(summary['largest_jump_interval'],[29,30])
            self.assertAlmostEqual(summary['largest_delta_HSL_nm'],-600.)
            self.assertEqual(summary['first_negative_overlap'][:2],[31,2])
            self.assertEqual(summary['overlap_order'],'after accepted FE event')
            self.assertEqual(summary['persistence']['classification'],'persists')
            self.assertEqual(summary['missing_native_steps'],0)
            self.assertFalse(summary['causal_claim'])
            stage=summary['anchor_chronology']
            self.assertEqual(stage['before_FE']['calcium_consumed'],30*1.e-7)
            self.assertEqual(stage['before_FE']['raw_membrane_calcium'],30*2.e-7)
            self.assertEqual(stage['after_FE']['next_step_cell_HSL_after_advection'],400.)
            self.assertEqual(len(stage['during_FE']['native_Newton_history']),2)

    def test_missing_native_sample_and_missing_stage_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            instruction=make_stage_fixture(tmp);observer=Path(tmp)/'observer'
            (observer/'frames/event_00000080_rank0001.npz').unlink()
            path=observer/'frames/event_00000030_rank0001.npz'
            with np.load(str(path)) as archive:frame={k:archive[k] for k in archive.files if k!='stage__before_kinetics__y'}
            np.savez_compressed(str(path),**frame)
            causal.analyze(str(instruction),str(Path(tmp)/'analysis'),plots=False)
            s=json.loads((Path(tmp)/'analysis/event_summaries.json').read_text())[0]
            self.assertEqual(s['missing_native_steps'],1);self.assertEqual(s['missing_stage_rows'],1)
            self.assertIn('unresolved',s['persistence']['classification'])

    def test_wrong_point_id_and_timestamp_offset_are_rejected(self):
        for key,value in [('global_ids',np.asarray([92])),('accepted_time',.0315)]:
            with tempfile.TemporaryDirectory() as tmp:
                instruction=make_stage_fixture(tmp);path=Path(tmp)/'observer/frames/event_00000030_rank0001.npz'
                with np.load(str(path)) as archive:frame={k:archive[k] for k in archive.files}
                frame[key]=value;np.savez_compressed(str(path),**frame)
                with self.assertRaises(ValueError):causal.analyze(str(instruction),str(Path(tmp)/'analysis'),plots=False)

    def test_persistence_rules_require_complete_native_followup(self):
        self.assertEqual(causal.persistence([(0,1),(1,.4)],1)['classification'].split(':')[0],'unresolved')
        values=[(i,1. if i==0 else .4 if i<20 else 1.2) for i in range(52)]
        self.assertEqual(causal.persistence(values,1)['classification'],'reversed')


class IntegrityAndComparisonTests(unittest.TestCase):
    def test_initial_evidence_is_read_only_and_mismatched_initial_states_are_rejected(self):
        sys.path.insert(0,str(ROOT/'python_codes'))
        runner=load('initial_capture_test',ROOT/'validation/run_fe_cycle_case.py')
        comparison=load('initial_compare_test',ROOT/'validation/compare_fe_cycle_smoke.py')
        values=np.asarray([.1,.2,.3])
        function=types.SimpleNamespace(vector=lambda:types.SimpleNamespace(get_local=lambda:values),
            function_space=lambda:types.SimpleNamespace(ufl_element=lambda:'synthetic Vector CG2'))
        hs=types.SimpleNamespace(data={'hs_length':1000.},
            myof=types.SimpleNamespace(y=np.asarray([.2,.3,.5,.8,.2]),data={'k_force':.001},implementation={'kinetic_scheme':'fixture','filament_compliance_factor':.5}),
            memb=types.SimpleNamespace(y=np.asarray([0.,.001]),data={'Ca_cytosol':0.},implementation={'scheme':'fixture'}))
        simulation=types.SimpleNamespace(mesh=types.SimpleNamespace(model={'functions':{'w':function}}),
            hs_objs_list=[hs],dofmap=np.asarray([91]),circ=types.SimpleNamespace(data={'v':[100.]}),
            hr=types.SimpleNamespace(data={'t_RR':.1}),br=[],comm=types.SimpleNamespace(Get_rank=lambda:0,allgather=lambda x:[x]))
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';a.mkdir();b.mkdir()
            runner.capture_initial_state(simulation,str(a));runner.capture_initial_state(simulation,str(b))
            np.testing.assert_array_equal(values,[.1,.2,.3])
            self.assertTrue(comparison.compare_initial(str(a),str(b))['exact_initial_evidence_equal'])
            path=b/'initial_rank0000.npz'
            with np.load(str(path),allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
            data['myof__k_force']=np.asarray([.01]);np.savez_compressed(str(path),**data)
            with self.assertRaises(ValueError):comparison.compare_initial(str(a),str(b))

    def test_storage_estimate_counts_all_routine_stage_arrays(self):
        prepare=load('followup_storage',ROOT/'validation/prepare_fe_cycle_case.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mesh=root/'mesh';mesh.write_bytes(b'synthetic hash-only fixture');image=root/'image';image.write_bytes(b'not executable')
            baseline=json.loads((ROOT/'demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json').read_text())
            baseline['mesh']['relative_path']=[False];baseline['mesh']['mesh_path']=[str(mesh)]
            path=root/'baseline.json';path.write_text(json.dumps(baseline))
            template=ROOT/'demos/ECM/sim_input/base_instruction_Br_ECM_30cycles_FE_diagnostic.json'
            prepare.prepare(str(path),str(template),str(root/'case'),'smoke',str(ROOT),str(image))
            estimate=json.loads((root/'case/storage_estimate.json').read_text())
            self.assertEqual(estimate['routine_full_field_frames'],2400)
            self.assertEqual(estimate['checkpoint_frames'],601)
            self.assertGreater(estimate['routine_field_payload_bytes'],5008*64*8)
            self.assertGreater(estimate['no_event_lower_planning_bytes'],170*1024**3)

    def test_original_driver_loop_stops_at_30_real_HR_intervals_and_flushes_closes(self):
        from lib2to3.refactor import RefactoringTool,get_fixers_from_package
        tool=RefactoringTool(get_fixers_from_package('lib2to3.fixes'))
        path=ROOT/'python_codes/LV_simulation/LV_simulation.py'
        tree=ast.parse(str(tool.refactor_string(path.read_text(),str(path))))
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='LV_simulation')
        method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='run_simulation')
        position=next(i for i,node in enumerate(method.body) if isinstance(node,ast.For) and isinstance(node.iter,ast.Call)
            and isinstance(node.iter.func,ast.Attribute) and node.iter.func.attr=='arange')
        module=ast.Module(body=method.body[position:]);module.type_ignores=[]
        hr_module=load('driver_hr',ROOT/'python_codes/LV_simulation/heart_rate/heart_rate.py')
        sys.path.insert(0,str(ROOT/'python_codes/LV_simulation/dependencies'))
        from fe_cycle_core import CycleLedger
        hr=hr_module.heart_rate(dict(t_active_period=[.003],t_quiescent_period=[.92],t_first_activation=[.1]))
        ledger=CycleLedger(30);writer=types.SimpleNamespace(closed=False)
        writer.close=lambda:setattr(writer,'closed',True)
        parent=types.SimpleNamespace(data={'time':0.,'new_beat':0},t_counter=0,comm=types.SimpleNamespace(allreduce=lambda value,op:value,Get_rank=lambda:0),
            prot=types.SimpleNamespace(data={'time_step':.001,'no_of_time_steps':60000}),output_profiling=False,optimized_output=True,
            solution_mesh=writer,flush_steps=[],marked=False)
        controller=types.SimpleNamespace(stop=False,begin=lambda:None)
        def step(dt):
            activation,new_beat,ed=hr.implement_time_step(dt)
            parent.data['new_beat']=new_beat;parent.data['time']+=dt;parent.t_counter+=1
            if len(ledger.cycles)==10:hr.data['t_quiescent_period']=.71
        def end():controller.stop=ledger.observe(parent.data['time'],parent.data['new_beat'])
        controller.end=end;parent._fe_cycle=controller;parent.implement_time_step=step
        parent.handle_output=lambda output:parent.flush_steps.append(parent.t_counter)
        parent._mark_buffered_output_complete=lambda:setattr(parent,'marked',True)
        scope=dict(self=parent,np=np,MPI=types.SimpleNamespace(SUM=0),output_struct={})
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
        self.assertEqual(len(ledger.cycles),30)
        self.assertEqual(parent.data['time'],ledger.cycles[-1]['end'])
        self.assertLess(parent.t_counter,60000)
        self.assertEqual(parent.flush_steps[-1],parent.t_counter)
        self.assertTrue(writer.closed);self.assertTrue(parent.marked)

    def integrity_fixture(self,root):
        import h5py
        root=Path(root);(root/'sim_output').mkdir();(root/'mesh_output').mkdir()
        case=dict(fe_cycle_diagnostic=dict(enabled=False,smoke_steps=1,directory=str(root/'observer')),
            output_handler=dict(output_data_path=[str(root/'sim_output/data.csv')],global_csv_frequency_n=[1],
                spatial_csv_frequency_n=[50],mesh_object_to_save=['displacement']))
        (root/'instruction.json').write_text(json.dumps(case));(root/'run_started.json').write_text(json.dumps({'ranks':1}))
        (root/'sim_output/data.csv').write_text('time,pressure\n0,80\n')
        (root/'sim_output/output_metadata.json').write_text(json.dumps(dict(complete=True,global_rows_written=1,spatial_rows_written=1)))
        with h5py.File(str(root/'final_mixed.h5'),'w') as handle:
            handle.create_dataset('w/vector',data=[0.]);handle.attrs['fixture']='HDF structural fixture, not a DOLFIN Function'
        (root/'mesh_output/solution.xdmf').write_text('<Xdmf><Domain><Grid><DataItem Format="HDF">../final_mixed.h5:/w/vector</DataItem></Grid></Domain></Xdmf>')
        return root/'instruction.json'

    def test_integrity_checks_reject_truncated_xml_and_offgrid_rows(self):
        module=load('integrity_test',ROOT/'validation/check_fe_cycle_output.py')
        with tempfile.TemporaryDirectory() as tmp:
            instruction=self.integrity_fixture(tmp)
            result=module.check(str(instruction))
            self.assertTrue(result['output_integrity_checks_passed'])
            self.assertFalse(result['full_pipeline_validated'])
            self.assertEqual(result['independent_FE_reconstruction'],{})
            (Path(tmp)/'mesh_output/solution.xdmf').write_text('<Xdmf>')
            with self.assertRaises(Exception):module.check(str(instruction))
        with tempfile.TemporaryDirectory() as tmp:
            instruction=self.integrity_fixture(tmp)
            (Path(tmp)/'sim_output/data.csv').write_text('time,pressure\n.001,80\n')
            with self.assertRaises(ValueError):module.check(str(instruction))

    def test_file_creation_does_not_substitute_for_reconstruction(self):
        module=load('integrity_no_FE',ROOT/'validation/check_fe_cycle_output.py')
        with tempfile.TemporaryDirectory() as tmp:
            instruction=self.integrity_fixture(tmp);case=json.loads(instruction.read_text());case['fe_cycle_diagnostic']['enabled']=True
            instruction.write_text(json.dumps(case));observer=Path(tmp)/'observer';observer.mkdir()
            (observer/'completion.json').write_text(json.dumps(dict(last_step=0,cycles=[],completed_cycles=0)))
            with self.assertRaises(FileNotFoundError):module.check(str(instruction))

    def test_invalid_hdf_signature_is_detected(self):
        module=load('integrity_hdf',ROOT/'validation/check_fe_cycle_output.py')
        with tempfile.TemporaryDirectory() as tmp:
            instruction=self.integrity_fixture(tmp);(Path(tmp)/'final_mixed.h5').write_bytes(b'truncated non-HDF data')
            with self.assertRaises(ValueError):module.check(str(instruction))

    def test_startup_geometry_is_one_row_not_a_decimated_time_history(self):
        module=load('integrity_static',ROOT/'validation/check_fe_cycle_output.py')
        with tempfile.TemporaryDirectory() as tmp:
            instruction=self.integrity_fixture(tmp);root=Path(tmp)
            case=json.loads(instruction.read_text());case['fe_cycle_diagnostic']['smoke_steps']=51
            instruction.write_text(json.dumps(case))
            (root/'sim_output/data.csv').write_text('time,pressure\n'+''.join('%.3f,80\n'%(step*.001) for step in range(51)))
            (root/'sim_output/lx_data.csv').write_text(',0,time\n0,1.5,0\n')
            (root/'sim_output/output_metadata.json').write_text(json.dumps(dict(complete=True,global_rows_written=51,spatial_rows_written=2)))
            np.save(str(root/'sim_output/quadrature_global_dof.npy'),np.asarray([0]))
            result=module.check(str(instruction))
            self.assertEqual(result['csv']['lx_data.csv']['rows'],1)
            self.assertEqual(result['csv']['lx_data.csv']['kind'],'reference_geometry')

    def test_native_comparator_checks_equilibrium_stress_and_Newton_counts(self):
        module=load('native_comparison',ROOT/'validation/compare_fe_cycle_smoke.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b';(a/'witness').mkdir(parents=True);(b/'observer').mkdir(parents=True)
            np.savez(str(a/'final_rank0000.npz'),fixture=np.asarray([0.]))
            result=dict(status='converged',iterations=0,residual_evaluations=1,records=[dict(iteration=0,
                residual_l2=0.,relative_residual=None,backend_converged=True,cumulative_linear_iterations=0,correction=None,trial=None)])
            for step in range(200):
                data=dict(accepted_time=(step+1)*.001,legacy_time=step*.001,global_ids=np.asarray([91]),
                    mixed_owned=np.asarray([.1]),active_equilibrium=np.asarray([10.]),newton_json=np.asarray(json.dumps(result)))
                for directory in (a/'witness',b/'observer'):np.savez(str(directory/('physical_%08d_rank0000.npz'%step)),**data)
            compared=module.compare_native(str(a),str(b));self.assertEqual(compared['frames'],200)
            self.assertEqual(compared['max_abs_physical_difference']['active_equilibrium'],0.)
            path=b/'observer/physical_00000030_rank0000.npz'
            with np.load(str(path)) as archive:data={k:archive[k] for k in archive.files}
            data['active_equilibrium']=np.asarray([11.]);np.savez(str(path),**data)
            with self.assertRaises(ValueError):module.compare_native(str(a),str(b))

    def test_native_build_refuses_wrong_platform_without_fabricating_output(self):
        module=load('native_build_test',ROOT/'validation/build_native_newton_observer.py')
        if sys.platform.startswith('linux'):self.skipTest('this specific test exercises the Windows refusal path')
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'build'
            with self.assertRaises(RuntimeError):module.build(str(target))
            self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
