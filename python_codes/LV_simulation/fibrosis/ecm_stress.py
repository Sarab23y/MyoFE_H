"""Diagnostic-only passive-ECM fiber-stress measurement."""
from __future__ import division
import numpy as np
from dolfin import Function, assemble, dx, inner, interpolate, project
from mpi4py import MPI
from ..dependencies.ecm_stress_diagnostic import positive_excess, relative_span, three_cycle_homeostatic_reference


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
        if self.cycles != 3 or self.activation_time <= 0 or self.tolerance <= 0:
            raise ValueError('Invalid ECM diagnostic calibration settings')
        qspace = self.mesh.model['function_spaces']['quadrature_space']
        count = len(Function(qspace).vector().get_local())
        self._ecm_integral, self._total_integral = np.zeros(count), np.zeros(count)
        self._duration, self._cycle_start = 0.0, 0.0
        self._scalar_integrals, self._completed = np.zeros(3), []
        self._frozen, self.homeostatic = False, np.zeros(count)
        self.data = {'ecm_reference_frozen': 0, 'ecm_reference_cycle_count': 0,
                     'ecm_reference_interval_start': np.nan, 'ecm_reference_interval_stop': np.nan,
                     'ecm_baseline_stable': 0, 'ecm_baseline_max_relative_change': np.nan}
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
        self._completed.append({'ecm': cycle.copy(), 'total': total.copy(), 'pressure': scalars[0],
                                'volume': scalars[1], 'heart_rate': scalars[2],
                                'start': self._cycle_start, 'stop': end_time})
        self._completed = self._completed[-self.cycles:]
        self._ecm_integral[:] = 0; self._total_integral[:] = 0; self._scalar_integrals[:] = 0
        self._duration, self._cycle_start = 0.0, end_time
        signed, positive = positive_excess(cycle, self.homeostatic)
        self._publish(self.data[self.FIELD_NAMES[0]], cycle, self.homeostatic, signed, positive, total)

    def _freeze(self):
        if len(self._completed) != self.cycles:
            raise RuntimeError('Fewer than three complete cycles before ECM calibration')
        records = self._completed
        self.homeostatic = three_cycle_homeostatic_reference([r['ecm'] for r in records])
        checks = [relative_span([r[key] for r in records]) for key in ('pressure', 'volume', 'heart_rate')]
        for previous, current in zip(records[:-1], records[1:]):
            num = self.comm.allreduce(float(np.sum((current['ecm']-previous['ecm'])**2)), op=MPI.SUM)
            den = self.comm.allreduce(float(np.sum(previous['ecm']**2)), op=MPI.SUM)
            checks.append(np.sqrt(num/max(den, np.finfo(float).eps)))
        maximum = max(checks); self.data['ecm_baseline_max_relative_change'] = maximum
        self.data['ecm_baseline_stable'] = int(maximum <= self.tolerance)
        if maximum > self.tolerance:
            raise RuntimeError('ECM baseline did not satisfy configured stability tolerance')
        self._frozen = True; self.data['ecm_reference_frozen'] = 1
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
