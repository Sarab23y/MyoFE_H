"""Opt-in observer of original execution stages. Never a coupled restart writer.

All projection results are separate Functions; original Functions are not set.
Backend Newton trial telemetry is explicitly unavailable on the opaque solve path.
"""
from __future__ import division, print_function
import json
import os
import time
import signal
import numpy as np
from .fe_cycle_core import CycleLedger, EventWindow, deformation_metrics, triggers, population_metrics


def clean_json(value):
    if isinstance(value, dict):
        return dict((str(k), clean_json(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class FECycleDiagnostic(object):
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.comm = parent.comm
        self.rank = self.comm.Get_rank()
        self.root = str(config['directory'])
        self.dt = float(parent.prot.data['time_step'])
        if abs(self.dt-0.001) > 1.e-15:
            raise ValueError('diagnostic case requires 1 ms timestep')
        self.ledger = CycleLedger(config['complete_cycles'])
        self.window = EventWindow(int(round(config['pre_event_s']/self.dt)),
                                  int(round(config['post_event_s']/self.dt)))
        self.previous = None
        self.last_step = -1
        self.fields = {}
        self.stages = {}
        self.event_steps = 0
        self.event_frames = 0
        self.checkpoint_frames = 0
        self.closed = False
        self.stop = False
        self.first_negative_overlap = None
        self.backend_nonconverged_solves = 0
        self.log_error = None
        self.created = time.time()
        self.reference_mass = np.asarray([np.sum(h.myof.y[:-2]) for h in parent.hs_objs_list])
        error = None
        if self.rank == 0:
            try:
                os.makedirs(self.root)  # exclusive: even an empty old dir is rejected
                os.mkdir(os.path.join(self.root, 'frames'))
            except Exception as exc:
                error = str(exc)
        error = self.comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)
        self.comm.Barrier()
        self.log = self.local_io(lambda: open(os.path.join(self.root, 'rank%04d.jsonl' % self.rank), 'w'))
        self.q = parent.mesh.model['function_spaces']['quadrature_space']
        self.functions = parent.mesh.model['functions']
        self.w = self.functions['w']
        self.mesh = parent.mesh.model['mesh']
        self._mapping()
        self._material_reference()
        from .native_newton_telemetry import NewtonTelemetry
        self.newton = NewtonTelemetry(
            lambda: self.w.sub(0, deepcopy=True).vector().get_local(),
            self.emit, self.comm, self.trial_metrics)
        for point, hs in enumerate(parent.hs_objs_list):
            def observe(stage, obj, detail=None, point=point):
                self.cellular(point, stage, obj, detail)
            hs._fe_observer = observe
        self.termination_signal = None
        self.previous_signal_handlers = {}
        def request_termination(number, unused_frame):
            # No I/O/MPI/exception inside a signal handler. Finish the existing
            # physical step, then use the original collective error/flush path.
            self.termination_signal = int(number)
        for name in ('SIGTERM','SIGINT','SIGUSR1'):
            number = getattr(signal,name,None)
            if number is not None:
                self.previous_signal_handlers[number] = signal.getsignal(number)
                signal.signal(number,request_termination)
        self.emit('capabilities', dict(
            checkpoint='mixed FE reconstruction only, NOT coupled restart',
            newton_trial_iterates=True, newton_corrections=True,
            newton_linear_iterations=True, newton_reason='forwarding native ELF hooks; requires runtime validation',
            hotspot_integer_ids_authoritative=False,
            historical_coordinates_required=True,
            timestamp='legacy_label=t; accepted_end=t+dt; same original update order'))

    def emit(self, kind, values):
        try:
            self.log.write(json.dumps(clean_json(dict(kind=kind,
                step=getattr(self, 'step', None), legacy_time=getattr(self, 'label', None),
                cardiac_cycle=getattr(self,'current_cycle',None),
                execution_stage=getattr(self,'current_stage',None),
                accepted_time=getattr(self, 'accepted_time', None), values=values)),
                allow_nan=False, sort_keys=True)+'\n')
            self.log.flush()
        except Exception as error:
            self.log_error = str(error)

    def local_io(self, callback):
        error = None
        result = None
        try:
            result = callback()
        except Exception as exc:
            error = 'rank %d: %s' % (self.rank, exc)
        errors = [value for value in self.comm.allgather(error) if value]
        if errors:
            raise RuntimeError('Diagnostic I/O failed: '+'; '.join(errors))
        return result

    def sync_log(self):
        errors = [value for value in self.comm.allgather(self.log_error) if value]
        if errors:
            raise RuntimeError('Diagnostic log write failed: '+'; '.join(errors))

    def collective_require(self,condition,message):
        errors=[value for value in self.comm.allgather(None if condition else
            'rank %d: %s'%(self.rank,message)) if value]
        if errors:raise RuntimeError('; '.join(errors))

    def project(self, expression, name):
        from .fenics_2017_compat import project_quadrature
        return project_quadrature(expression, self.q, 'cycle30_'+name,
                                  self.comm).vector().get_local().copy()

    def _mapping(self):
        from dolfin import assemble, TestFunction, dx
        from .buffered_output import validate_global_dof_partition
        mesh, q = self.mesh, self.q
        dm = q.dofmap()
        begin, end = dm.ownership_range()
        ids = np.asarray(dm.tabulate_local_to_global_dofs(), dtype=np.int64)
        self.ids = ids[:end-begin].copy()
        self.collective_require(len(self.ids)==len(self.parent.hs_objs_list),
                                'owned quadrature/Myosim length mismatch')
        self.collective_require(np.array_equal(self.ids,np.asarray(self.parent.dofmap)),
                                'production scalar IDs differ from owned quadrature IDs')
        all_ids = self.comm.allgather(self.ids.tolist())
        if not validate_global_dof_partition(all_ids, q.dim()):
            raise RuntimeError('missing or duplicate owned quadrature IDs')
        if hasattr(mesh,'init_global'):
            mesh.init_global(0)
            mesh.init_global(3)
        self.reference_weights = assemble(TestFunction(q)*dx(domain=mesh,
            metadata={'quadrature_degree':2,'quadrature_scheme':'default'}),
            form_compiler_parameters={'representation':'quadrature'}).get_local()
        if not all(self.comm.allgather(bool(np.all(np.isfinite(self.reference_weights)) and
                                              np.all(self.reference_weights > 0)))):
            raise RuntimeError('invalid diagnostic reference quadrature weights')
        self.coordinates = q.tabulate_dof_coordinates().reshape((-1, 3))[:len(self.ids)].copy()
        global_cells = mesh.topology().global_indices(3)
        global_vertices = mesh.topology().global_indices(0)
        self.collective_require(len(global_cells)==mesh.num_cells() and len(global_vertices)==mesh.num_vertices(),
                                'global mesh entity indices unavailable; cannot certify geometry')
        cells = np.full(len(self.ids), -1, dtype=np.int64)
        local_q = cells.copy()
        cell_records = {}
        cell_conflict=False
        for cell, vertices in enumerate(mesh.cells()):
            global_cell = int(global_cells[cell])
            cell_records[global_cell] = [int(global_vertices[v]) for v in vertices]
            for ordinal, local in enumerate(dm.cell_dofs(cell)):
                if local < len(self.ids):
                    if cells[local] >= 0 and cells[local] != global_cell:
                        cell_conflict=True
                    cells[local], local_q[local] = global_cell, ordinal
        self.collective_require(not cell_conflict,'Quadrature point assigned to multiple cells')
        self.collective_require(not np.any(cells<0),'unlocated owned quadrature points')
        all_cells = {}
        for records in self.comm.allgather(cell_records):
            for key, value in records.items():
                if key in all_cells and all_cells[key] != value:
                    raise RuntimeError('global cell connectivity disagrees between ranks')
                all_cells[key] = value
        faces = {}
        import itertools
        for cell, vertices in all_cells.items():
            for face in itertools.combinations(vertices, 3):
                faces.setdefault(tuple(sorted(face)), []).append(cell)
        neighbors = dict((cell, set()) for cell in all_cells)
        for shared in faces.values():
            for cell in shared:
                neighbors[cell].update(other for other in shared if other != cell)
        self.cells = cells
        self.local_q = local_q
        ranges = self.comm.allgather(self.w.vector().local_range())
        f0 = self.functions['f0']
        self.fiber = np.column_stack([self.project(f0[i], 'f0_'+str(i)) for i in range(3)])
        self.hsl0 = self.project(self.functions['hsl0'], 'reference_hsl')
        self.local_io(lambda: np.savez(os.path.join(self.root, 'mapping_rank%04d.npz' % self.rank),
                 global_ids=self.ids, coordinates=self.coordinates, global_cells=cells,
                 local_quadrature=local_q, fiber=self.fiber, hsl0=self.hsl0,
                 reference_volume_weights=self.reference_weights,
                 connectivity=np.asarray([all_cells[int(c)] for c in cells])))
        W = self.w.function_space()
        self.local_io(lambda: np.savez(os.path.join(self.root, 'mixed_mapping_rank%04d.npz' % self.rank),
            local_to_global=np.asarray(W.dofmap().tabulate_local_to_global_dofs(),dtype=np.int64),
            owned_range=np.asarray(self.w.vector().local_range(),dtype=np.int64),
            global_cells=np.asarray(global_cells,dtype=np.int64),
            mixed_cell_local_dofs=np.asarray([W.dofmap().cell_dofs(c) for c in range(mesh.num_cells())]),
            displacement_cell_parent_local_dofs=np.asarray([W.sub(0).dofmap().cell_dofs(c) for c in range(mesh.num_cells())]),
            pressure_cell_parent_local_dofs=np.asarray([W.sub(1).dofmap().cell_dofs(c) for c in range(mesh.num_cells())])))
        def write_mapping():
            if self.rank != 0:
                return
            with open(os.path.join(self.root, 'mapping.json'), 'w') as handle:
                json.dump(clean_json(dict(world_size=self.comm.Get_size(),
                    cells=all_cells, face_neighbors=dict((k, sorted(v)) for k,v in neighbors.items()),
                    scalar_owned_partition_verified=True,
                    mixed_component_mapping_verified=False,
                    historical_id_correspondence_verified=False,
                    function_space=str(self.w.function_space().ufl_element()),
                    quadrature_space=str(q.ufl_element()),
                    checkpoint_dof_ranges=ranges
                    )), handle, indent=2)
        self.local_io(write_mapping)
        self.tracked = np.zeros(len(self.ids), dtype=bool)
        reference = self.config.get('hotspot_coordinates')
        if reference:
            reference = np.asarray(reference, dtype=float)
            matched_cells = []
            for coordinate in reference:
                matches = np.where(np.linalg.norm(self.coordinates-coordinate, axis=1) < 1.e-9)[0]
                total = sum(self.comm.allgather(len(matches)))
                if total != 1:
                    raise RuntimeError('historical hotspot coordinate is not uniquely mapped')
                matched_cells.extend(self.cells[matches].tolist())
            selected = set(c for part in self.comm.allgather(matched_cells) for c in part)
            selected.update(n for c in list(selected) for n in neighbors[c])
            self.tracked = np.asarray([int(c) in selected for c in self.cells])
            self.emit('hotspot_mapping', dict(coordinates=reference,
                owned_neighbor_ids=self.ids[self.tracked], historical_integer_ids_not_reused=True))

    def begin(self):
        self.current_stage='original_timestep_before_mechanics'
        self.step = int(self.parent.t_counter)
        self.label = float(self.parent.data['time'])
        self.accepted_time = self.label+self.dt
        self.stages = {}
        self.fields = {}

    def note_beat(self,new_beat):
        if self.ledger.start is None and not new_beat:
            self.current_cycle=0  # initialization, before the first genuine start
        else:
            self.current_cycle=len(self.ledger.cycles)+1+int(bool(new_beat and self.ledger.start is not None))

    def _material_reference(self):
        """Archive actual resolved passive coefficients, not just JSON defaults."""
        from dolfin import HDF5File, Function
        resolved = self.functions['dolfin_functions']['passive_params']
        leaves = {}
        constants = {}
        def visit(name, item):
            if isinstance(item, dict):
                for key, value in item.items():
                    visit(name+'__'+str(key) if name else str(key), value)
            else:
                candidate = item[-1] if isinstance(item, (list,tuple)) else item
                if hasattr(candidate,'vector') and hasattr(candidate,'function_space'):
                    leaves[name] = candidate
                elif hasattr(candidate,'values'):
                    constants[name] = np.asarray(candidate.values()).tolist()
                elif np.isscalar(candidate):
                    constants[name] = candidate
                else:
                    constants[name] = str(candidate)
        visit('',resolved)
        handle = HDF5File(self.mesh.mpi_comm(), str(os.path.join(self.root,'reference_coefficients.h5')), 'w')
        try:
            handle.write(self.mesh,'/mesh')
            for name in sorted(leaves):
                source = leaves[name]
                copy = Function(source.function_space())
                copy.vector().set_local(source.vector().get_local().copy())
                copy.vector().apply('insert')
                handle.write(copy,'/material_'+name)
        finally:
            handle.close()
        self.emit('reference_coefficients',dict(functions=sorted(leaves),constants=constants,
                  immutable_healthy_parameters_only=True))

    def cellular(self, point, stage, hs, detail):
        myof = hs.myof
        y = np.asarray(myof.y).copy()
        try:
            raw_overlap = myof.return_n_overlap()
            overlap_defined = True
        except Exception:
            # An extra read-only probe must not create a new physical failure
            # before advection changes HSL. Normal source calls remain untouched.
            raw_overlap = np.nan
            overlap_defined = False
        values = dict(y=y, hs_length=hs.data['hs_length'],
                      membrane_y=np.asarray(hs.memb.y).copy(),
                      calcium=hs.memb.data.get('Ca_cytosol'),
                      stored_cb=myof.cb_stress, stored_passive=myof.pas_stress,
                      raw_overlap=raw_overlap, raw_overlap_defined=overlap_defined,
                      stored_overlap=getattr(myof, 'n_overlap', None))
        for key, value in hs.memb.data.items():
            if np.isscalar(value) and not isinstance(value, (str, bytes)):
                values['membrane__'+key] = float(value)
        for key in ('k_1','k_3','k_on','k_cb','cb_number_density','x_ps'):
            if key in myof.data:
                values['parameter__'+key] = float(myof.data[key])
        if detail:
            values.update(detail)
        if stage == 'before_advection':
            delta = detail['delta_hsl']*myof.implementation['filament_compliance_factor']
            positions = myof.x-delta
            families = 2 if myof.implementation['kinetic_scheme'].startswith('4_state') else 1
            values['out_of_domain_interpolation_bins'] = int(families*np.sum(
                (positions < np.min(myof.x)) | (positions > np.max(myof.x)))) if delta else 0
        values.update(dict((k, v[0]) for k,v in population_metrics(
            y.reshape((1,-1)), self.reference_mass[point]).items()))
        if myof.implementation['kinetic_scheme'].startswith('3_state'):
            values['M_FG'] = float(np.sum(y[2:-2]))
        else:
            values['M_PRE'] = float(np.sum(y[2:2+myof.no_of_x_bins]))
            values['M_POST'] = float(np.sum(y[2+myof.no_of_x_bins:-2]))
        self.stages.setdefault(stage, {})[point] = values

    def before_solve(self):
        self.sync_log()
        self.w_before = self.w.vector().get_local().copy()
        self.u_before = self.w.sub(0, deepcopy=True).vector().get_local().copy()
        self.residual_initial = self.residual_norm()
        self.active_initial = self.project(self.functions['cb_stress'],'active_newton_initial')
        self.emit('solve_begin', dict(residual_l2=self.residual_initial,
                                      cycle_completed=len(self.ledger.cycles)))
        self.solve_started = time.time()
        self.newton.start()
        self.current_stage='original_mechanics_solve'

    def trial_metrics(self):
        F = self.parent.mesh.model['uflforms'].Fmat()
        tensors = np.column_stack([self.project(F[i,j], 'trial_F%d%d'%(i,j))
            for i in range(3) for j in range(3)]).reshape((-1,3,3))
        J = np.linalg.det(tensors)
        stretch = np.linalg.norm(np.einsum('nij,nj->ni',tensors,self.fiber),axis=1)
        j_index=int(np.argmin(J));s_index=int(np.argmin(stretch))
        local = (float(np.min(J)),float(np.max(J)),float(np.min(stretch)),
                 int(self.ids[j_index]),int(self.ids[s_index]),self.rank,
                 int(np.sum(~np.isfinite(J) | ~np.isfinite(stretch))))
        values = self.comm.allgather(local)
        if any(v[6] for v in values):
            return dict(J_min=None,J_max=None,stretch_min=None,extrema_defined=False,
                        nonfinite_points=sum(v[6] for v in values))
        worst_j=min(values,key=lambda v:(v[0],v[3]));worst_s=min(values,key=lambda v:(v[2],v[4]))
        return dict(J_min=min(v[0] for v in values),J_max=max(v[1] for v in values),
                    stretch_min=min(v[2] for v in values),extrema_defined=True,
                    J_min_global_id=worst_j[3],J_min_rank=worst_j[5],
                    stretch_min_global_id=worst_s[4],stretch_min_rank=worst_s[5])

    def residual_norm(self):
        from dolfin import assemble
        params = self.parent.solver.parameters
        b = assemble(params['Ftotal'], form_compiler_parameters={'representation':'uflacs'})
        for bc in params['boundary_conditions']:
            bc.apply(b, self.w.vector())
        return b.norm('l2')

    def accepted(self):
        returned_at=time.time()
        self.current_stage='post_solve_observation'
        self.sync_log()
        self.newton_result = self.newton.finish(returned=True)
        self.backend_nonconverged_solves += int(self.newton_result['status']!='converged')
        from dolfin import inner
        from .fenics_2017_compat import mixed_pressure_quadrature
        forms = self.parent.mesh.model['uflforms']
        F = forms.Fmat()
        arrays = [self.project(F[i, j], 'F%d%d' % (i,j))
                  for i in range(3) for j in range(3)]
        Ft = np.column_stack(arrays).reshape((-1, 3, 3))
        self.fields = deformation_metrics(Ft, self.fiber, self.hsl0)
        self.fields['F'] = Ft
        self.fields['hsl0'] = self.hsl0.copy()
        self.fields['active_newton_initial_PK2_coefficient'] = self.active_initial.copy()
        self.fields['hsl_old_pre_reset'] = self.project(self.functions['hsl_old'], 'pre_reset_hsl')
        f0 = self.functions['f0']
        for name, label in (('ground_matrix_passive','ground'),
                            ('collagen_passive','collagen'),('myo_passive_PK2','Xi')):
            self.fields[label+'_PK2_ff'] = self.project(
                inner(f0, self.functions[name]*f0), name+'_ff')
        # These projections occur BEFORE the original hsl_old reset.
        self.fields['active_equilibrium_PK2_coefficient'] = self.project(
            self.functions['cb_stress'], 'active_equilibrium')
        p = mixed_pressure_quadrature(self.w, self.q, 'cycle30_pressure', self.comm).vector().get_local()
        inverse_factor = np.full(len(Ft), np.nan)
        for point, tensor in enumerate(Ft):
            try:
                inverse_factor[point] = np.dot(self.fiber[point],
                    np.linalg.solve(np.dot(tensor.T, tensor), self.fiber[point]))
            except np.linalg.LinAlgError:
                pass  # undefined diagnostic stays NaN; never repairs a deformation
        self.fields['pressure_legacy_PK2_ff'] = -p*inverse_factor
        self.fields['pressure_equilibrium_PK2_ff'] = -p*self.fields['J']*inverse_factor
        # Preserve fiber norms: active scalar coefficient != ff contraction for nonunit f0.
        self.fields['active_equilibrium_PK2_ff'] = (
            self.fields['active_equilibrium_PK2_coefficient']*self.fields['fiber_norm']**4)
        self.fields['active_newton_initial_PK2_ff'] = self.active_initial*self.fields['fiber_norm']**4
        self.fields['ECM_PK2_ff'] = self.fields['ground_PK2_ff']+self.fields['collagen_PK2_ff']
        self.fields['total_equilibrium_PK2_ff'] = (self.fields['ECM_PK2_ff']+
            self.fields['Xi_PK2_ff']+self.fields['active_equilibrium_PK2_ff']+
            self.fields['pressure_equilibrium_PK2_ff'])
        final = self.residual_norm()
        net = self.w.vector().get_local()-self.w_before
        local_max = float(np.max(np.abs(net))) if len(net) else 0.
        u_change = self.w.sub(0, deepcopy=True).vector().get_local()-self.u_before
        displacement_change = float(np.max(np.abs(u_change))) if len(u_change) else 0.
        self.emit('solve_returned', dict(final_residual_l2=final,
            relative_residual=(final/self.residual_initial if self.residual_initial else None),
            elapsed_s=returned_at-self.solve_started,
            elapsed_includes_native_callbacks=True,
            accepted_projection_wall_s=time.time()-returned_at,
            max_net_mixed_coefficient_change=max(self.comm.allgather(local_max)),
            max_net_displacement_coefficient_change=max(self.comm.allgather(displacement_change)),
            net_change_is_not_newton_correction=True,
            nonlinear_status=self.newton_result['status'],
            iterations=self.newton_result['iterations'],
            accepted_state=self.trial_metrics(),
            state_category=('accepted_converged_solver_return' if self.newton_result['status']=='converged'
                            else 'driver_returned_nonconverged_state'),
            linear_status='native cumulative iteration count recorded; LU reason code unavailable'))

    def end(self):
        self.current_stage='end_timestep_observation'
        self.sync_log()
        if self.config.get('smoke_steps'):
            from .fe_invariance_witness import capture_step
            capture_step(self,self.root,self.fields['active_equilibrium_PK2_coefficient'],self.newton_result)
        self.fields['active_output_PK2_coefficient'] = self.project(
            self.functions['cb_stress'], 'active_output')
        self.fields['active_stage_difference'] = (self.fields['active_output_PK2_coefficient']-
            self.fields['active_equilibrium_PK2_coefficient'])
        self.fields['active_output_PK2_ff'] = self.fields['active_output_PK2_coefficient']*self.fields['fiber_norm']**4
        self.fields['hsl_FE_production_list'] = np.asarray(self.parent.hs_length_list).copy()
        self.fields['stretch_FE_production_data'] = np.asarray(self.parent.data['fiber_stretch_raw']).copy()
        hs_objects = self.parent.hs_objs_list
        self.fields['overlap'] = np.asarray([obj.myof.n_overlap for obj in hs_objects])
        self.fields['overlap_cellular_hsl'] = np.asarray([obj.data['hs_length'] for obj in hs_objects])
        self.fields['min_bin'] = np.asarray([np.min(obj.myof.y) for obj in hs_objects])
        self.fields.update(population_metrics(np.asarray([obj.myof.y for obj in hs_objects]), self.reference_mass))
        self.fields['calcium_output'] = np.asarray([obj.memb.data['Ca_cytosol'] for obj in hs_objects])
        self.fields['hsl_old_post_reset'] = self.project(self.functions['hsl_old'], 'post_reset_hsl')
        self.fields['Sff_Xi_coefficient'] = self.project(self.functions['Sff'], 'Xi_coefficient')
        masks = triggers(self.fields, self.previous, self.config['thresholds'])
        reasons = dict((key, self.ids[value].tolist()) for key, value in masks.items() if np.any(value))
        owned=set(self.ids.tolist());global_newton=[]
        correction_limit=self.config['thresholds'].get('newton_displacement_correction_max')
        for record in self.newton_result['records']:
            trial=record.get('trial') or {}
            for metric,limit,identity,reason in (
                ('J_min',self.config['thresholds']['J'],'J_min_global_id','trial_low_J'),
                ('stretch_min',self.config['thresholds']['stretch'],'stretch_min_global_id','trial_low_stretch')):
                if trial.get(metric) is not None and trial[metric]<limit and trial.get(identity) in owned:
                    reasons.setdefault(reason,[]).append(trial[identity])
            correction=record.get('correction')
            if correction_limit is not None and correction and correction['applied_displacement_max']>correction_limit:
                global_newton.append(record['iteration'])
        reasons=dict((key,sorted(set(value))) for key,value in reasons.items())
        if self.newton_result['status']!='converged':
            reasons['newton_nonconvergence']=self.ids.tolist()
        any_event = any(self.comm.allgather(bool(reasons) or bool(global_newton)))
        if any_event:
            self.event_steps += 1
            self.emit('event', reasons)
            if global_newton and self.rank==0:
                self.emit('global_newton_event',dict(iterations=global_newton,
                    reason='applied displacement correction exceeds configured mesh-coordinate threshold',
                    quadrature_point_identity='global event; displacement DOFs are not quadrature IDs'))
        if np.any(self.tracked):
            selected = np.where(self.tracked)[0]
            self.emit('hotspots_and_neighbors', dict(global_ids=self.ids[selected],
                fields=dict((k,v[selected]) for k,v in self.fields.items()),
                stages=dict((stage, dict((p, values[p]) for p in selected))
                            for stage,values in self.stages.items())))
        stretch = self.fields['stretch']
        local_summary = dict(J_min=float(np.min(self.fields['J'])),
            J_max=float(np.max(self.fields['J'])), stretch_min=float(np.min(stretch)),
            stretch_max=float(np.max(stretch)), sigma_min=float(np.min(self.fields['sigma_min'])),
            inverted=int(np.sum(self.fields['J'] <= 0)),
            low_J=int(np.sum(self.fields['J'] < self.config['thresholds']['J'])),
            negative_overlap_points=int(np.sum(self.fields['overlap'] < 0)),
            negative_population_points=int(np.sum(self.fields['min_bin'] < 0)),
            negative_population_bins=int(np.sum(self.fields['negative_bins'])),
            signed_deformed_volume_q2=float(np.dot(self.reference_weights, self.fields['J'])),
            max_delta_stretch=(float(np.max(np.abs(stretch-self.previous))) if self.previous is not None else 0.))
        summaries = self.comm.allgather(local_summary)
        global_summary = dict(J_min=min(v['J_min'] for v in summaries),
            J_max=max(v['J_max'] for v in summaries),
            stretch_min=min(v['stretch_min'] for v in summaries),
            stretch_max=max(v['stretch_max'] for v in summaries),
            sigma_min=min(v['sigma_min'] for v in summaries),
            inverted=sum(v['inverted'] for v in summaries),
            low_J=sum(v['low_J'] for v in summaries),
            negative_overlap_points=sum(v['negative_overlap_points'] for v in summaries),
            negative_population_points=sum(v['negative_population_points'] for v in summaries),
            negative_population_bins=sum(v['negative_population_bins'] for v in summaries),
            signed_deformed_volume_q2=sum(v['signed_deformed_volume_q2'] for v in summaries),
            max_delta_stretch=max(v['max_delta_stretch'] for v in summaries))
        if global_summary['negative_overlap_points'] and self.first_negative_overlap is None:
            self.first_negative_overlap = self.accepted_time
        volumes, pressures, flows = self.parent.return_system_values()
        if self.rank == 0:
            self.emit('global', dict(mechanics=global_summary, volumes=volumes,
                pressures=pressures, flows=flows, heart_rate=self.parent.hr.return_heart_rate(),
                activation=self.parent.hr.data['activation'], RR=self.parent.hr.data['t_RR'],
                baroreflex=self.parent.br.data if self.parent.br else None))
        # Copy all original coefficients; no original vector is changed on replay.
        vector_names = ['w','f0','s0','n0','hsl0','hsl_old','y_vec']
        vector_names.extend(name for name in ('cb_number_density','k_cb','x_ps',
            'k_1','k_3','k_on','k_act','k_serca') if name in self.functions and
            hasattr(self.functions[name],'vector') and hasattr(self.functions[name],'function_space'))
        vectors = dict((name, self.functions[name].vector().get_local().copy()) for name in vector_names)
        dense_stages = {}
        for stage, points in self.stages.items():
            if len(points) != len(self.ids):
                raise RuntimeError('missing cellular stage records: '+stage)
            for name in points[0]:
                dense_stages[stage+'__'+name] = np.asarray([
                    points[p][name] if points[p][name] is not None else np.nan
                    for p in range(len(self.ids))])
        frame = dict(fields=dict((k, v.copy()) for k,v in self.fields.items()),
                     vectors=vectors, stages=dense_stages, label=self.label,
                     accepted_time=self.accepted_time)
        selected = self.window.add(self.step, frame, any_event)
        periodic = self.step % int(self.config['checkpoint_stride']) == 0
        emitted_current = False
        for number, value in selected:
            self.write_frame(number, value, event=True)
            emitted_current |= number == self.step
        if periodic and not emitted_current:
            self.write_frame(self.step, frame, event=False)
        elif self.step % int(self.config['full_field_stride']) == 0 and not emitted_current:
            self.write_arrays(self.step, frame, 'field')
        self.previous = stretch.copy()
        self.last_step = self.step
        self.stop = self.ledger.observe(self.accepted_time, self.parent.data['new_beat'])
        self.emit('cycle_ledger', dict(completed=len(self.ledger.cycles),
                                      newest=self.ledger.cycles[-1:] ))
        smoke_steps = int(self.config.get('smoke_steps', 0))
        if smoke_steps and self.step+1 >= smoke_steps:
            self.stop = True
        abort=int(self.config.get('diagnostic_abort_after_steps',0))
        if abort and smoke_steps and self.step+1>=abort:
            raise RuntimeError('intentional diagnostic smoke interruption at accepted timestep %d'%(self.step+1))
        pending = [value for value in self.comm.allgather(self.termination_signal) if value is not None]
        if pending:
            raise RuntimeError('diagnostic termination requested at accepted-step boundary: signals %s' % pending)

    def write_arrays(self, step, frame, prefix):
        self.local_io(lambda: self._write_arrays_local(step, frame, prefix))

    def _write_arrays_local(self, step, frame, prefix):
        path = os.path.join(self.root, 'frames', '%s_%08d_rank%04d.npz' % (prefix,step,self.rank))
        payload = dict(frame['fields'])
        payload.update(global_ids=self.ids, legacy_time=frame['label'],
                       accepted_time=frame['accepted_time'])
        payload.update(dict(('stage__'+key, value) for key,value in frame['stages'].items()))
        # open exclusively; a duplicate is a bug, never an append/overwrite.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, 'wb') as handle:
            np.savez_compressed(handle, **payload)

    def write_frame(self, step, frame, event):
        from dolfin import HDF5File, Function
        self.write_arrays(step, frame, 'event' if event else 'checkpoint')
        path = os.path.join(self.root, 'frames', 'state_%08d.h5' % step)
        exists = self.comm.bcast(os.path.exists(path) if self.rank == 0 else None, root=0)
        if exists:
            # A periodic frame already exists before a later event selects it.
            # Its exact state remains available; avoid rewriting the HDF5 file.
            return
        handle = HDF5File(self.mesh.mpi_comm(), str(path), 'w')
        error = None
        try:
            handle.write(self.mesh, '/mesh')
            for name, values in frame['vectors'].items():
                copy = Function(self.functions[name].function_space())
                copy.vector().set_local(values)
                copy.vector().apply('insert')
                handle.write(copy, '/'+name)
        finally:
            handle.close()
        # Reopen now and verify all owned mixed coefficients bit-for-bit.
        reader = HDF5File(self.mesh.mpi_comm(), str(path), 'r')
        try:
            recovered = Function(self.w.function_space())
            reader.read(recovered, '/w')
            same = np.array_equal(recovered.vector().get_local(), frame['vectors']['w'])
        finally:
            reader.close()
        if not all(self.comm.allgather(bool(same))):
            raise RuntimeError('mixed CG2 checkpoint round-trip mismatch')
        self.checkpoint_frames += 1
        self.event_frames += int(event)
        self.emit('checkpoint_verified', dict(frame_step=step, path=path,
            legacy_time=frame['label'], accepted_time=frame['accepted_time'],
            same_partition_exact_mixed_vector=True, full_coupled_restart=False))

    def finish(self, success=False, error=None):
        if self.closed:
            return
        if hasattr(self,'newton') and self.newton.active:
            try:self.newton.finish(returned=False)
            except RuntimeError as diagnostic_error:
                if error is None:raise
                self.emit('telemetry_error_during_original_failure',dict(
                    original_error=error,telemetry_error=str(diagnostic_error)))
        if error:
            # Preserve only already accepted history; a failed iterate is never
            # mislabeled as a converged event frame. No post-failure states exist.
            for number, frame in self.window.history:
                if number > self.window.last_written:
                    self.write_frame(number, frame, event=True)
                    self.window.last_written = number
        self.emit('finalization', dict(success=success, error=error))
        self.sync_log()
        self.log.close()
        self.closed = True
        # Manifest deliberately cannot certify unavailable smoke/solver telemetry.
        if self.rank == 0:
            manifest = dict(completed_cycles=len(self.ledger.cycles), cycles=self.ledger.cycles,
                target_cycles=self.ledger.target, last_step=self.last_step,
                simulation_completed=bool(success and len(self.ledger.cycles)==self.ledger.target),
                smoke_only=bool(self.config.get('smoke_steps')),
                observer_buffers_flushed=True, legacy_buffers_flushed=bool(success),
                error=error, event_steps=self.event_steps,
                failure_execution_stage=getattr(self,'current_stage',None) if error else None,
                missing_post_event_steps=self.window.unfinished(self.last_step),
                first_negative_overlap_accepted_time=self.first_negative_overlap,
                backend_nonconverged_solves=self.backend_nonconverged_solves,
                ecm_calibration_schedule_s=40.,
                calibration_time_reached=bool(getattr(self,'label',-1) >= 40.),
                ecm_reference_calibrated='inspect original ECM diagnostic report; time alone is insufficient',
                full_coupled_restart=False, full_pipeline_validated=False,
                elapsed_s=time.time()-self.created,
                unavailable=['actual container validation of native Newton hooks',
                             'verified historical hotspot coordinate correspondence',
                             'executed 1/32-rank invariance and mapping evidence'])
            temporary = os.path.join(self.root, 'completion.tmp')
            with open(temporary, 'w') as handle:
                json.dump(clean_json(manifest), handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(temporary, os.path.join(self.root, 'completion.json'))
        for number, handler in getattr(self,'previous_signal_handlers',{}).items():
            signal.signal(number,handler)
        if hasattr(self,'newton'):
            self.newton.close()
