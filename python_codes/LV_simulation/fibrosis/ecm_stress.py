"""Diagnostic-only passive-ECM fiber-stress measurement."""
from __future__ import division
import json
import os
import sys
import numpy as np
from dolfin import Function, assemble, dx, inner, interpolate, project
from mpi4py import MPI
from ..dependencies.ecm_stress_diagnostic import (positive_excess,
    three_cycle_homeostatic_reference)
from ..dependencies.ecm_stability import calibration_decision, stability_results


class ECMStressDiagnostic(object):
    """Measure fraction-weighted ECM PK2 fiber stress without remodeling."""
    FIELD_NAMES = ('ecm_instantaneous_fiber_stress', 'ecm_cycle_averaged_fiber_stress',
                   'ecm_homeostatic_fiber_stress', 'ecm_signed_stress_deviation',
                   'ecm_positive_excess_stress', 'ecm_cycle_averaged_total_fiber_stress')

    def __init__(self, config, parent):
        self.parent, self.mesh, self.comm = parent, parent.mesh, parent.comm
        self.activation_time = float(config['activation_time'][0])
        self.cycles = int(config.get('homeostatic_cycles', [3])[0])
        self.tolerance = float(config.get('baseline_stability_relative_tolerance', [0.05])[0])
        self.stability_mode = str(config.get('baseline_stability_mode', ['strict'])[0]).lower()
        if (self.cycles != 3 or self.activation_time <= 0 or self.tolerance <= 0 or
                self.stability_mode not in ('strict', 'warn')):
            raise ValueError('Invalid ECM diagnostic calibration settings')
        qspace = self.mesh.model['function_spaces']['quadrature_space']
        count = len(Function(qspace).vector().get_local())
        self._ecm_integral, self._total_integral = np.zeros(count), np.zeros(count)
        self._duration, self._cycle_start = 0.0, 0.0
        self._scalar_integrals, self._completed = np.zeros(3), []
        self._frozen, self.homeostatic = False, np.zeros(count)
        self.data = {'ecm_reference_frozen': 0, 'ecm_reference_cycle_count': 0,
                     'ecm_reference_interval_start': np.nan, 'ecm_reference_interval_stop': np.nan,
                     'ecm_baseline_stable': 0, 'ecm_reference_provisional': 0,
                     'ecm_excess_valid': 0,
                     'ecm_baseline_failed_criterion_count': 0,
                     'ecm_baseline_max_relative_change': np.nan}
        passive = self.mesh.model['functions']['dolfin_functions']['passive_params']
        self._initial_fractions = {}
        for name in ('phi_m', 'phi_g', 'phi_c'):
            values = passive[name][-1].vector().get_local().copy()
            self._initial_fractions[name] = values
            self.data['ecm_' + name + '_mean'] = self._global_stats(values)[0]
            # Reuse the authoritative passive-parameter Function for XDMF;
            # the diagnostic never assigns to it.
            self.mesh.model['functions'][name] = passive[name][-1]
        self.data['ecm_reference_myocardial_volume'] = float(assemble(
            1.0*dx(domain=self.mesh.model['mesh']),
            form_compiler_parameters={'representation': 'uflacs'}))
        self.data['ecm_fraction_max_change'] = 0.0
        for name in self.FIELD_NAMES:
            self.data[name] = np.zeros(count)
            fn = Function(qspace); fn.rename(name, '')
            self.mesh.model['functions'][name] = fn
        zeros = np.zeros(count)
        self._publish(zeros, zeros, zeros, zeros, zeros, zeros)

    def _fiber_stresses(self):
        f = self.mesh.model['functions']; f0 = f['f0']
        # These are already fraction-weighted PK2 tensors; Xi, active and
        # constraint-pressure stresses are deliberately excluded.
        expressions = (inner(f0, (f['ground_matrix_passive'] + f['collagen_passive'])*f0),
                       inner(f0, f['total_stress']*f0))
        values = []
        for expression in expressions:
            dg = project(expression, self.mesh.model['function_spaces']['growth_scalar_FS'],
                         form_compiler_parameters={'representation': 'uflacs'})
            q = interpolate(dg, self.mesh.model['function_spaces']['quadrature_space'])
            values.append(q.vector().get_local())
        return values

    def _global_stats(self, values):
        n = self.comm.allreduce(len(values), op=MPI.SUM)
        return (self.comm.allreduce(float(np.sum(values)), op=MPI.SUM)/n,
                self.comm.allreduce(float(np.min(values)), op=MPI.MIN),
                self.comm.allreduce(float(np.max(values)), op=MPI.MAX))

    def _publish(self, instantaneous, cycle, homeostatic, signed, positive, total_cycle):
        for name, values in zip(self.FIELD_NAMES, (instantaneous, cycle, homeostatic, signed, positive, total_cycle)):
            values = np.asarray(values); self.data[name] = values.copy()
            fn = self.mesh.model['functions'][name]; fn.vector().set_local(values); fn.vector().apply('insert')
        for prefix, values in (('ecm_cycle_averaged', cycle), ('ecm_homeostatic', homeostatic),
                               ('ecm_signed_deviation', signed), ('ecm_positive_excess', positive)):
            mean, minimum, maximum = self._global_stats(values)
            self.data[prefix + '_mean'], self.data[prefix + '_min'], self.data[prefix + '_max'] = mean, minimum, maximum
        mean, minimum, maximum = self._global_stats(total_cycle)
        self.data['ecm_cycle_averaged_total_fiber_stress_mean'] = mean
        self.data['ecm_cycle_averaged_total_fiber_stress_min'] = minimum
        self.data['ecm_cycle_averaged_total_fiber_stress_max'] = maximum

    def _finish_cycle(self, end_time):
        if self._duration <= 0: return
        cycle, total, scalars = self._ecm_integral/self._duration, self._total_integral/self._duration, self._scalar_integrals/self._duration
        mean, minimum, maximum = self._global_stats(cycle)
        l2 = self._field_l2(cycle)
        self._completed.append({'ecm': cycle.copy(), 'total': total.copy(), 'pressure': scalars[0],
                                'volume': scalars[1], 'heart_rate': scalars[2],
                                'start': self._cycle_start, 'stop': end_time,
                                'duration': self._duration, 'ecm_min': minimum,
                                'ecm_mean': mean, 'ecm_max': maximum,
                                'ecm_l2': l2})
        self._completed = self._completed[-self.cycles:]
        self._ecm_integral[:] = 0; self._total_integral[:] = 0; self._scalar_integrals[:] = 0
        self._duration, self._cycle_start = 0.0, end_time
        if self._frozen:
            signed, positive = positive_excess(cycle, self.homeostatic)
        else:
            # Zero-valued placeholders keep PETSc/XDMF output finite.  The
            # validity flag prevents these from being interpreted as excess.
            signed, positive = np.zeros_like(cycle), np.zeros_like(cycle)
        self._publish(self.data[self.FIELD_NAMES[0]], cycle, self.homeostatic, signed, positive, total)

    def _field_l2(self, values):
        """Physical L2 norm using the FE integration measure (global assemble)."""
        qspace = self.mesh.model['function_spaces']['quadrature_space']
        fn = Function(qspace)
        fn.vector().set_local(np.asarray(values)); fn.vector().apply('insert')
        squared = float(assemble(fn*fn*dx(domain=self.mesh.model['mesh']),
                                 form_compiler_parameters={'representation': 'uflacs'}))
        return np.sqrt(max(squared, 0.0))

    def _field_relative_change(self, previous, current):
        qspace = self.mesh.model['function_spaces']['quadrature_space']
        old, new = Function(qspace), Function(qspace)
        old.vector().set_local(previous); old.vector().apply('insert')
        new.vector().set_local(current); new.vector().apply('insert')
        domain = self.mesh.model['mesh']
        num = float(assemble((new-old)*(new-old)*dx(domain=domain),
                             form_compiler_parameters={'representation': 'uflacs'}))
        old2 = float(assemble(old*old*dx(domain=domain),
                              form_compiler_parameters={'representation': 'uflacs'}))
        new2 = float(assemble(new*new*dx(domain=domain),
                              form_compiler_parameters={'representation': 'uflacs'}))
        scale = max(old2, new2)
        if scale == 0.0:
            return 0.0
        return np.sqrt(max(num, 0.0)/scale)

    def _write_stability_report(self, report):
        if self.comm.Get_rank() != 0:
            return
        print '\n[ECMBaselineStability]'
        for index, record in enumerate(report['cycles']):
            print ('  Cycle %d: start=%.6g s stop=%.6g s duration=%.6g s '
                   'pressure=%.9g volume=%.9g HR=%.9g ECM[min/mean/max/L2]='
                   '%.9g/%.9g/%.9g/%.9g' %
                   (index+1, record['start'], record['stop'], record['duration'],
                    record['pressure'], record['volume'], record['heart_rate'],
                    record['ecm_min'], record['ecm_mean'], record['ecm_max'],
                    record['ecm_l2']))
        print '  Consecutive-cycle stability (tolerance %.3f%%):' % (100.0*self.tolerance)
        for pair in report['pairs']:
            print '    Cycle %d -> %d' % (pair['previous_cycle']+1, pair['current_cycle']+1)
            for key in ('pressure', 'volume', 'heart_rate', 'ecm_stress_field'):
                value = pair['changes'][key]
                print '      %-16s %12.6f%%  %s' % (key, 100.0*value,
                                                    'PASS' if pair['passed'][key] else 'FAIL')
        print '  Combined decision: %s' % ('PASS' if report['stable'] else 'FAIL')
        if report['failed_criteria']:
            print '  Failed criteria: %s' % ', '.join(report['failed_criteria'])
        if report['reference_provisional']:
            if 'ecm_stress_field' in report['failed_criteria']:
                print ('  WARNING: ECM stress field exceeds the %.3f%% stability '
                       'tolerance.' % (100.0*self.tolerance))
            print ('  WARNING: baseline stability tolerance exceeded (%s). '
                   'A provisional ECM reference was frozen for diagnostic-only use.' %
                   ', '.join(report['failed_criteria']))
        sys.stdout.flush(); sys.stderr.flush()
        output_path = self.parent.instruction_data['output_handler']['output_data_path'][0]
        directory = os.path.dirname(output_path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, 'ecm_baseline_stability.json'), 'w') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    def _field_change_hotspots(self, previous, current, count=10):
        """Return the largest owned pointwise changes with global DOF IDs."""
        difference = np.abs(np.asarray(current)-np.asarray(previous))
        if not np.all(np.isfinite(difference)):
            raise RuntimeError('Nonfinite ECM pointwise field difference')
        local_count = min(int(count), len(difference))
        indices = np.argsort(difference)[-local_count:]
        dofmap = self.mesh.model['function_spaces']['quadrature_space'].dofmap()
        local = []
        for index in indices:
            local.append({'global_quadrature_dof': int(dofmap.local_to_global_index(int(index))),
                          'previous': float(previous[index]),
                          'current': float(current[index]),
                          'absolute_change': float(difference[index])})
        gathered = self.comm.gather(local, root=0)
        if self.comm.Get_rank() != 0:
            return []
        combined = [item for rank_items in gathered for item in rank_items]
        return sorted(combined, key=lambda item: item['absolute_change'], reverse=True)[:count]

    def _freeze(self):
        if len(self._completed) != self.cycles:
            raise RuntimeError('Fewer than three complete cycles before ECM calibration')
        records = self._completed
        for record in records:
            if (len(record['ecm']) != len(self.homeostatic) or
                    not np.all(np.isfinite(record['ecm']))):
                raise RuntimeError('Invalid or mismatched ECM cycle field before calibration')
        candidate = three_cycle_homeostatic_reference([r['ecm'] for r in records])
        field_changes = [self._field_relative_change(previous['ecm'], current['ecm'])
                         for previous, current in zip(records[:-1], records[1:])]
        pairs = stability_results(records, field_changes, self.tolerance)
        checks = [value for pair in pairs for value in pair['changes'].values()]
        maximum = max(checks); self.data['ecm_baseline_max_relative_change'] = maximum
        self.data['ecm_baseline_stable'] = int(maximum <= self.tolerance)
        failed = sorted(set(key for pair in pairs for key, passed in pair['passed'].items()
                            if not passed))
        self.data['ecm_baseline_failed_criterion_count'] = len(failed)
        report_cycles = []
        for record in records:
            report_cycles.append(dict((key, float(record[key])) for key in
                ('start', 'stop', 'duration', 'pressure', 'volume', 'heart_rate',
                 'ecm_min', 'ecm_mean', 'ecm_max', 'ecm_l2')))
        decision = calibration_decision(failed, self.stability_mode)
        provisional = decision['provisional']
        hotspots = []
        for pair_index, (previous, current) in enumerate(zip(records[:-1], records[1:])):
            hotspots.append({'previous_cycle': pair_index,
                             'current_cycle': pair_index+1,
                             'points': self._field_change_hotspots(previous['ecm'], current['ecm'])})
        report = {'activation_time': self.activation_time,
                  'tolerance': self.tolerance, 'cycles': report_cycles,
                  'pairs': pairs, 'stable': not failed,
                  'failed_criteria': failed, 'stability_mode': self.stability_mode,
                  'reference_frozen': decision['freeze'],
                  'reference_provisional': provisional,
                  'field_change_hotspots': hotspots if self.comm.Get_rank() == 0 else []}
        self._write_stability_report(report)
        if decision['raise']:
            raise RuntimeError('ECM baseline stability failed: %s' % ', '.join(failed))
        self.homeostatic = candidate
        self._frozen = True; self.data['ecm_reference_frozen'] = 1
        self.data['ecm_reference_provisional'] = int(provisional)
        self.data['ecm_excess_valid'] = 1
        self.data['ecm_reference_cycle_count'] = self.cycles
        self.data['ecm_reference_interval_start'], self.data['ecm_reference_interval_stop'] = records[0]['start'], records[-1]['stop']
        signed, positive = positive_excess(self.data[self.FIELD_NAMES[1]], self.homeostatic)
        self._publish(self.data[self.FIELD_NAMES[0]], self.data[self.FIELD_NAMES[1]], self.homeostatic,
                      signed, positive, self.data[self.FIELD_NAMES[5]])

    def record_time_step(self, time_step, end_diastolic):
        now = float(self.parent.data['time'])
        if not self._frozen and now >= self.activation_time: self._freeze()
        ecm, total = self._fiber_stresses()
        if len(ecm) != len(self._ecm_integral): raise RuntimeError('ECM quadrature ordering/size changed')
        dt = float(time_step); self._ecm_integral += ecm*dt; self._total_integral += total*dt; self._duration += dt
        self._scalar_integrals += dt*np.asarray((self.parent.circ.data['p'][-1], self.parent.circ.data['v'][-1], self.parent.data['heart_rate']))
        self.data[self.FIELD_NAMES[0]] = ecm.copy()
        fn = self.mesh.model['functions'][self.FIELD_NAMES[0]]; fn.vector().set_local(ecm); fn.vector().apply('insert')
        if end_diastolic: self._finish_cycle(now)
        if end_diastolic:
            passive = self.mesh.model['functions']['dolfin_functions']['passive_params']
            change = 0.0
            for name, initial in self._initial_fractions.items():
                current = passive[name][-1].vector().get_local()
                change = max(change, float(np.max(np.abs(current-initial))))
            self.data['ecm_fraction_max_change'] = self.comm.allreduce(
                change, op=MPI.MAX)
            if self.data['ecm_fraction_max_change'] > 1.0e-12:
                raise RuntimeError('ECM diagnostic composition changed unexpectedly')
