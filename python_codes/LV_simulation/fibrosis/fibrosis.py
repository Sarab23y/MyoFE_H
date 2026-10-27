# -*- coding: utf-8 -*-
"""Optional stress-driven, volume-neutral fibrosis composition exchange."""
from __future__ import division

import numpy as np
from dolfin import Function, assemble, dx, inner, interpolate, project
from mpi4py import MPI

from ..dependencies.constituent_mass import ConstituentMassBookkeeper
from ..dependencies.fibrosis_evolution import (
    calibrated_stress_references, effective_fibrosis_stimulus,
    exact_collagen_update,
    exact_collagen_update_from_signal, stress_range_signal,
    validate_fibrosis_config,
    validate_fraction_compatibility)


class StressDrivenFibrosis(object):
    """Cycle-average total fiber stress and exchange matrix for collagen."""

    def __init__(self, config, parent):
        self.config = validate_fibrosis_config(config)
        if not self.config['enabled']:
            raise ValueError('Cannot initialize disabled fibrosis model')
        if 'growth' in parent.instruction_data['model']:
            raise ValueError(
                'Volume-neutral fibrosis cannot run with geometric growth')
        self.parent = parent
        self.comm = parent.comm
        self.mesh = parent.mesh
        self.data = {}
        passive = self.mesh.model['functions']['dolfin_functions'][
            'passive_params']
        fractions = dict((name, passive[name][-1].vector().get_local().tolist())
                         for name in ('phi_m', 'phi_g', 'phi_c'))
        point_count = len(fractions['phi_m'])
        self.bookkeeper = ConstituentMassBookkeeper(
            fractions, point_count,
            {'enabled': [True], 'density': [self.config['density']],
             'positive_growth_allocation': {'mode': ['proportional']}})
        validate_fraction_compatibility(
            fractions['phi_m'], fractions['phi_g'], fractions['phi_c'],
            self.config['phi_c_max'])
        self._stress_integral = np.zeros(point_count)
        self._cycle_duration = 0.0
        self._homeostatic_samples = []
        self._baseline_cycle_samples = []
        calibration = self.config.get(
            'stress_reference_calibration', {'enabled': False})
        self._auto_reference_calibration = calibration['enabled']
        self._stress_references_frozen = not self._auto_reference_calibration
        self._remodeling_stress_integral = np.zeros(point_count)
        self._remodeling_duration = 0.0
        self._remodeling_cycles = 0
        self.homeostatic_stress = None
        self.activation_start = 0.0
        self.activation_stop = float('inf')
        self.event_number = 0
        self.initial_reference_volume = self._reference_volume()
        self.initial_global_masses = self._global_masses()
        self._initialize_data(point_count)
        self._apply_composition()
        self._publish_stress_diagnostics(
            np.zeros(point_count), np.zeros(point_count),
            np.zeros(point_count))

    def _fraction_functions(self):
        return self.mesh.model['functions']['dolfin_functions'][
            'passive_params']

    def _reference_volume(self):
        return float(assemble(
            1.0*dx(domain=self.mesh.model['mesh']),
            form_compiler_parameters={'representation': 'uflacs'}))

    def _total_fiber_stress(self):
        functions = self.mesh.model['functions']
        expression = inner(
            functions['f0'], functions['total_stress']*functions['f0'])
        if 'growth_scalar_FS' in self.mesh.model['function_spaces']:
            projected = project(
                expression,
                self.mesh.model['function_spaces']['growth_scalar_FS'],
                form_compiler_parameters={'representation': 'uflacs'})
            projected = interpolate(
                projected,
                self.mesh.model['function_spaces']['quadrature_space'])
        else:
            projected = project(
                expression,
                self.mesh.model['function_spaces']['quadrature_space'],
                form_compiler_parameters={'representation': 'uflacs'})
        return projected.vector().get_local()

    def _initialize_data(self, point_count):
        zeros = np.zeros(point_count)
        self.data['fib_cycle_averaged_total_fiber_stress'] = zeros.copy()
        self.data['fib_homeostatic_fiber_stress'] = zeros.copy()
        self.data['fib_normalized_stimulus'] = zeros.copy()
        self.data['fib_effective_stimulus'] = zeros.copy()
        self.data['fib_stress_range_normalized'] = zeros.copy()
        self.data['fib_signal'] = zeros.copy()
        self.data['fib_gamma_c'] = self.config['gamma_c']
        self.data['fib_tau_c'] = self.config['tau_c']
        if self.config['stimulus'] == \
                'cycle_averaged_total_fiber_stress_range':
            if self._stress_references_frozen:
                self.data['fib_S_ff_min'] = self.config['S_ff_min']
                self.data['fib_S_ff_max'] = self.config['S_ff_max']
            else:
                self.data['fib_S_ff_min'] = np.nan
                self.data['fib_S_ff_max'] = np.nan
                calibration = self.config['stress_reference_calibration']
                self.data['fib_calibration_cycles_collected'] = 0
                self.data['fib_calibration_point_count'] = 0
                self.data['fib_calibration_observed_min'] = np.nan
                self.data['fib_calibration_observed_max'] = np.nan
                self.data['fib_calibration_lower_percentile'] = \
                    calibration['lower_percentile']
                self.data['fib_calibration_upper_percentile'] = \
                    calibration['upper_percentile']
                self.data['fib_baseline_signal_mean'] = np.nan
                self.data['fib_baseline_signal_min'] = np.nan
                self.data['fib_baseline_signal_max'] = np.nan
        self.data['fib_reference_calibrated'] = \
            1 if self._stress_references_frozen else 0
        fractions = self.bookkeeper.fractions()
        self.data['fib_phi_m'] = np.asarray(fractions['phi_m'])
        self.data['fib_phi_g'] = np.asarray(fractions['phi_g'])
        self.data['fib_phi_c'] = np.asarray(fractions['phi_c'])
        self.data['fib_active'] = 0
        self.data['fib_event_number'] = 0
        self.data['fib_remodeling_timestep'] = 0.0
        self.data['fib_reference_volume'] = self.initial_reference_volume
        self.data['fib_reference_volume_residual'] = 0.0
        self._update_global_diagnostics(zeros, zeros)

    def _global_stats(self, values):
        values = np.asarray(values)
        local_count = len(values)
        total_count = self.comm.allreduce(local_count, op=MPI.SUM)
        total_sum = self.comm.allreduce(float(np.sum(values)), op=MPI.SUM)
        local_min = float(np.min(values)) if local_count else float('inf')
        local_max = float(np.max(values)) if local_count else float('-inf')
        return (total_sum/total_count,
                self.comm.allreduce(local_min, op=MPI.MIN),
                self.comm.allreduce(local_max, op=MPI.MAX))

    def _global_masses(self):
        densities = self.bookkeeper.referential_density
        masses = {}
        for constituent in ('myofiber', 'ground_matrix', 'collagen'):
            function = Function(
                self.mesh.model['function_spaces']['quadrature_space'])
            function.vector().set_local(np.asarray(densities[constituent]))
            function.vector().apply('insert')
            masses[constituent] = float(assemble(
                function*dx(domain=self.mesh.model['mesh']),
                form_compiler_parameters={'representation': 'uflacs'}))
        masses['total'] = sum(masses.values())
        return masses

    def _calculate_stimuli(self, cycle_stress):
        if self.config['stimulus'] == \
                'cycle_averaged_total_fiber_stress_range':
            normalized, signal = stress_range_signal(
                cycle_stress, self.config['S_ff_min'],
                self.config['S_ff_max'], self.config['gamma_c'],
                self.config['tau_c'])
            normalized = np.asarray(normalized)
            signal = np.asarray(signal)
            effective = signal*self.config['tau_c']
            return normalized, effective, signal
        denominator = np.maximum(
            np.abs(self.homeostatic_stress), self.config['stress_epsilon'])
        normalized = np.maximum(
            (cycle_stress - self.homeostatic_stress)/denominator, 0.0)
        effective = np.asarray(effective_fibrosis_stimulus(
            normalized, self.config['gamma_c']))
        return normalized, effective, effective/self.config['tau_c']

    def _global_bound_fraction(self, values, bound):
        values = np.asarray(values)
        local_count = len(values)
        total_count = self.comm.allreduce(local_count, op=MPI.SUM)
        local_hits = int(np.count_nonzero(values == bound))
        total_hits = self.comm.allreduce(local_hits, op=MPI.SUM)
        return float(total_hits)/float(total_count)

    def _update_global_diagnostics(self, cycle_stress, stimulus,
                                   effective_stimulus=None, signal=None):
        if effective_stimulus is None:
            effective_stimulus = np.asarray(
                effective_fibrosis_stimulus(
                    stimulus, self.config['gamma_c']))
        if signal is None:
            signal = effective_stimulus/self.config['tau_c']
        fractions = self.bookkeeper.fractions()
        for name, values in (
                ('cycle_averaged_total_fiber_stress', cycle_stress),
                ('homeostatic_fiber_stress',
                 self.homeostatic_stress if self.homeostatic_stress is not None
                 else np.zeros(len(cycle_stress))),
                ('normalized_stimulus', stimulus),
                ('effective_stimulus', effective_stimulus),
                ('stress_range_normalized', stimulus),
                ('signal', signal),
                ('phi_m', fractions['phi_m']),
                ('phi_g', fractions['phi_g']),
                ('phi_c', fractions['phi_c'])):
            mean, minimum, maximum = self._global_stats(values)
            self.data['fib_%s_mean' % name] = mean
            self.data['fib_%s_min' % name] = minimum
            self.data['fib_%s_max' % name] = maximum
        if self.config['stimulus'] == \
                'cycle_averaged_total_fiber_stress_range':
            self.data['fib_lower_bound_fraction'] = \
                self._global_bound_fraction(stimulus, 0.0)
            self.data['fib_upper_bound_fraction'] = \
                self._global_bound_fraction(stimulus, 1.0)
        else:
            self.data['fib_lower_bound_fraction'] = 0.0
            self.data['fib_upper_bound_fraction'] = 0.0
        masses = self._global_masses()
        for name, value in masses.items():
            self.data['fib_global_mass_' + name] = value
        self.data['fib_total_mass_residual'] = \
            masses['total'] - self.initial_global_masses['total']

    def _apply_composition(self):
        fractions = self.bookkeeper.fractions()
        passive = self._fraction_functions()
        for key in ('phi_m', 'phi_g', 'phi_c'):
            function = passive[key][-1]
            function.vector().set_local(np.asarray(fractions[key]))
            function.vector().apply('insert')
            self.mesh.model['functions'][key] = function
            self.data['fib_' + key] = np.asarray(fractions[key])
        density_names = {
            'myofiber': 'constituent_density_myofiber',
            'ground_matrix': 'constituent_density_ground_matrix',
            'collagen': 'constituent_density_collagen',
        }
        for constituent, name in density_names.items():
            function = Function(
                self.mesh.model['function_spaces']['quadrature_space'])
            values = np.asarray(
                self.bookkeeper.referential_density[constituent])
            function.vector().set_local(values)
            function.vector().apply('insert')
            self.mesh.model['functions'][name] = function
            self.data['fib_density_' + constituent] = values
        total = Function(self.mesh.model['function_spaces']['quadrature_space'])
        total.vector().set_local(
            self.data['fib_density_myofiber'] +
            self.data['fib_density_ground_matrix'] +
            self.data['fib_density_collagen'])
        total.vector().apply('insert')
        self.mesh.model['functions']['constituent_density_total'] = total
        self.data['fib_density_total'] = total.vector().get_local()

    def _set_diagnostic_function(self, name, values):
        function = self.mesh.model['functions'].get(name)
        if function is None:
            function = Function(
                self.mesh.model['function_spaces']['quadrature_space'])
        function.vector().set_local(np.asarray(values))
        function.vector().apply('insert')
        self.mesh.model['functions'][name] = function

    def _publish_stress_diagnostics(self, cycle_stress, stimulus,
                                    effective_stimulus=None, signal=None):
        if effective_stimulus is None:
            effective_stimulus = np.asarray(
                effective_fibrosis_stimulus(
                    stimulus, self.config['gamma_c']))
        if signal is None:
            signal = effective_stimulus/self.config['tau_c']
        homeostatic = self.homeostatic_stress
        if homeostatic is None:
            homeostatic = np.zeros(len(cycle_stress))
        self._set_diagnostic_function(
            'fibrosis_cycle_averaged_total_fiber_stress', cycle_stress)
        self._set_diagnostic_function(
            'fibrosis_homeostatic_fiber_stress', homeostatic)
        self._set_diagnostic_function('fibrosis_normalized_stimulus', stimulus)
        self._set_diagnostic_function(
            'fibrosis_effective_stimulus', effective_stimulus)
        self._set_diagnostic_function(
            'fibrosis_stress_range_normalized', stimulus)
        self._set_diagnostic_function('fibrosis_signal', signal)
        if self.config['stimulus'] == \
                'cycle_averaged_total_fiber_stress_range':
            point_count = len(cycle_stress)
            lower = self.config.get('S_ff_min', 0.0)
            upper = self.config.get('S_ff_max', 0.0)
            self._set_diagnostic_function(
                'fibrosis_S_ff_min', np.full(point_count, lower))
            self._set_diagnostic_function(
                'fibrosis_S_ff_max', np.full(point_count, upper))

    def _record_baseline_cycle(self, cycle_average):
        """Retain only the requested complete cycles before activation."""
        cycles = self.config['stress_reference_calibration']['cycles']
        self._baseline_cycle_samples.append(cycle_average.copy())
        if len(self._baseline_cycle_samples) > cycles:
            self._baseline_cycle_samples.pop(0)
        self.data['fib_calibration_cycles_collected'] = \
            len(self._baseline_cycle_samples)
        self.data['fib_cycle_averaged_total_fiber_stress'] = \
            cycle_average.copy()
        zeros = np.zeros(len(cycle_average))
        self._update_global_diagnostics(cycle_average, zeros, zeros, zeros)
        self._publish_stress_diagnostics(cycle_average, zeros, zeros, zeros)

    def _freeze_stress_references(self):
        """Freeze robust global references from pre-activation cycles.

        Per-point cycle averages are first averaged over the retained complete
        cycles. Exact global spatial percentiles are then evaluated on rank 0
        from owned local Quadrature-vector entries and broadcast to all ranks.
        """
        calibration = self.config['stress_reference_calibration']
        cycles = calibration['cycles']
        if len(self._baseline_cycle_samples) != cycles:
            raise RuntimeError(
                'Automatic stress calibration requires %d complete '
                'pre-activation cycles; found %d' %
                (cycles, len(self._baseline_cycle_samples)))
        baseline = np.mean(self._baseline_cycle_samples, axis=0)
        # All ranks enter the gather even when one rank contains invalid data;
        # rank 0 reports the error and broadcasts one failure to every rank.
        gathered = self.comm.gather(baseline, root=0)
        result = None
        if self.comm.Get_rank() == 0:
            try:
                global_values = np.concatenate(gathered)
                if len(global_values) == 0 or \
                        not np.all(np.isfinite(global_values)):
                    raise RuntimeError(
                        'Global baseline stress is empty or nonfinite')
                lower, upper, observed_min, observed_max = \
                    calibrated_stress_references(
                        global_values, calibration['lower_percentile'],
                        calibration['upper_percentile'])
                result = (True, '', lower, upper,
                          int(len(global_values)), observed_min, observed_max)
            except Exception as error:
                result = (False, str(error), 0.0, 0.0, 0, 0.0, 0.0)
        result = self.comm.bcast(result, root=0)
        if not result[0]:
            raise RuntimeError(
                'Automatic stress calibration failed: ' + result[1])
        lower, upper, count, observed_min, observed_max = result[2:]
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise RuntimeError(
                'Automatic stress references are nonfinite or degenerate')
        self.config['S_ff_min'] = lower
        self.config['S_ff_max'] = upper
        self._stress_references_frozen = True
        self.homeostatic_stress = baseline.copy()
        self.data['fib_homeostatic_fiber_stress'] = baseline.copy()
        self.data['fib_S_ff_min'] = lower
        self.data['fib_S_ff_max'] = upper
        self.data['fib_reference_calibrated'] = 1
        self.data['fib_calibration_point_count'] = count
        self.data['fib_calibration_observed_min'] = observed_min
        self.data['fib_calibration_observed_max'] = observed_max
        self.data['fib_calibration_lower_percentile'] = \
            calibration['lower_percentile']
        self.data['fib_calibration_upper_percentile'] = \
            calibration['upper_percentile']
        normalized, effective, signal = self._calculate_stimuli(baseline)
        mean, minimum, maximum = self._global_stats(signal)
        self.data['fib_baseline_signal_mean'] = mean
        self.data['fib_baseline_signal_min'] = minimum
        self.data['fib_baseline_signal_max'] = maximum
        self._update_global_diagnostics(
            baseline, normalized, effective, signal)
        self._publish_stress_diagnostics(
            baseline, normalized, effective, signal)
        if self.comm.Get_rank() == 0:
            print '[StressDrivenFibrosis] fixed stress references'
            print '  percentile method:', \
                calibration['lower_percentile'], \
                calibration['upper_percentile']
            print '  S_ff_min/S_ff_max:', lower, upper
            print '  observed min/max:', observed_min, observed_max
            print '  baseline signal min/mean/max:', minimum, mean, maximum

    def set_activation_window(self, growth_activations):
        """Use protocol.growth as the fibrosis remodeling schedule only."""
        if len(growth_activations) == 0:
            self.activation_start = self.config['activation_time']
            self.activation_stop = float('inf')
            return
        if len(growth_activations) != 1:
            raise ValueError(
                'Volume-neutral fibrosis requires exactly one '
                'protocol.growth activation')
        activation = growth_activations[0].data
        self.activation_start = float(activation['t_start_s'])
        self.activation_stop = float(activation['t_stop_s'])
        if self.activation_stop <= self.activation_start:
            raise ValueError('Fibrosis activation stop must follow start')

    def record_time_step(self, time_step, end_diastolic):
        current_time = self.parent.data['time']
        active = self.activation_start <= current_time < self.activation_stop
        self.data['fib_active'] = 1 if active else 0
        if self._auto_reference_calibration and not active:
            if current_time >= self.activation_start:
                return
            stress = self._total_fiber_stress()
            self._stress_integral += stress*time_step
            self._cycle_duration += time_step
            if end_diastolic:
                if self._cycle_duration <= 0.0:
                    raise RuntimeError(
                        'Fibrosis calibration cycle duration is not positive')
                cycle_average = self._stress_integral/self._cycle_duration
                self._stress_integral[:] = 0.0
                self._cycle_duration = 0.0
                self._record_baseline_cycle(cycle_average)
            return
        if not active:
            return
        if self._auto_reference_calibration and \
                not self._stress_references_frozen:
            self._freeze_stress_references()
            # Exclude the partial baseline cycle crossing activation time.
            self._stress_integral[:] = 0.0
            self._cycle_duration = 0.0
        stress = self._total_fiber_stress()
        self._stress_integral += stress*time_step
        self._cycle_duration += time_step
        if not end_diastolic:
            return
        if self._cycle_duration <= 0.0:
            raise RuntimeError('Fibrosis cycle duration is not positive')
        cycle_average = self._stress_integral/self._cycle_duration
        cycle_duration = self._cycle_duration
        self._stress_integral[:] = 0.0
        self._cycle_duration = 0.0

        if self.homeostatic_stress is None:
            self._homeostatic_samples.append(cycle_average.copy())
            if len(self._homeostatic_samples) == \
                    self.config['homeostatic_cycles']:
                self.homeostatic_stress = np.mean(
                    self._homeostatic_samples, axis=0)
                self.data['fib_homeostatic_fiber_stress'] = \
                    self.homeostatic_stress.copy()
            self.data['fib_cycle_averaged_total_fiber_stress'] = \
                cycle_average.copy()
            self._update_global_diagnostics(cycle_average,
                                             np.zeros(len(cycle_average)))
            self._publish_stress_diagnostics(
                cycle_average, np.zeros(len(cycle_average)))
            return

        self._remodeling_stress_integral += cycle_average*cycle_duration
        self._remodeling_duration += cycle_duration
        self._remodeling_cycles += 1
        cycle_stimulus, cycle_effective_stimulus, cycle_signal = \
            self._calculate_stimuli(cycle_average)
        self.data['fib_cycle_averaged_total_fiber_stress'] = \
            cycle_average.copy()
        self.data['fib_normalized_stimulus'] = cycle_stimulus.copy()
        self.data['fib_effective_stimulus'] = \
            cycle_effective_stimulus.copy()
        self.data['fib_stress_range_normalized'] = cycle_stimulus.copy()
        self.data['fib_signal'] = cycle_signal.copy()
        self._update_global_diagnostics(
            cycle_average, cycle_stimulus, cycle_effective_stimulus,
            cycle_signal)
        self._publish_stress_diagnostics(
            cycle_average, cycle_stimulus, cycle_effective_stimulus,
            cycle_signal)
        if self._remodeling_cycles < \
                self.config['remodeling_frequency_cycles']:
            return
        remodeling_average = self._remodeling_stress_integral / \
            self._remodeling_duration
        stimulus, effective_stimulus, signal = \
            self._calculate_stimuli(remodeling_average)
        fractions = self.bookkeeper.fractions()
        if self.config['stimulus'] == \
                'cycle_averaged_total_fiber_stress_range':
            phi_c_new = exact_collagen_update_from_signal(
                fractions['phi_c'], signal, self._remodeling_duration,
                self.config['phi_c_max'])
        else:
            phi_c_new = exact_collagen_update(
                fractions['phi_c'], stimulus, self._remodeling_duration,
                self.config['tau_c'], self.config['phi_c_max'],
                self.config['gamma_c'])
        before_total = np.asarray(
            [sum(self.bookkeeper.referential_density[name][point]
                 for name in ('myofiber', 'ground_matrix', 'collagen'))
             for point in range(len(phi_c_new))])
        self.bookkeeper.exchange_ground_matrix_for_collagen(phi_c_new)
        after_total = np.asarray(
            [sum(self.bookkeeper.referential_density[name][point]
                 for name in ('myofiber', 'ground_matrix', 'collagen'))
             for point in range(len(phi_c_new))])
        if np.max(np.abs(after_total - before_total)) > 1.0e-12:
            raise RuntimeError('Fibrosis changed total local density')
        self._apply_composition()
        self.event_number += 1
        self.data['fib_event_number'] = self.event_number
        self.data['fib_remodeling_timestep'] = self._remodeling_duration
        self.data['fib_cycle_averaged_total_fiber_stress'] = \
            remodeling_average.copy()
        self.data['fib_normalized_stimulus'] = stimulus.copy()
        self.data['fib_effective_stimulus'] = effective_stimulus.copy()
        self.data['fib_stress_range_normalized'] = stimulus.copy()
        self.data['fib_signal'] = signal.copy()
        current_volume = self._reference_volume()
        self.data['fib_reference_volume'] = current_volume
        self.data['fib_reference_volume_residual'] = \
            current_volume - self.initial_reference_volume
        self._update_global_diagnostics(
            remodeling_average, stimulus, effective_stimulus, signal)
        self._publish_stress_diagnostics(
            remodeling_average, stimulus, effective_stimulus, signal)
        tolerance = 1.0e-10*max(1.0, self.initial_reference_volume)
        if abs(self.data['fib_reference_volume_residual']) > tolerance:
            raise RuntimeError(
                'Fibrosis-only event changed myocardial reference volume')
        mass_tolerance = 1.0e-10*max(
            1.0, self.initial_global_masses['total'])
        if abs(self.data['fib_total_mass_residual']) > mass_tolerance:
            raise RuntimeError(
                'Fibrosis-only event changed total constituent mass')
        self._remodeling_stress_integral[:] = 0.0
        self._remodeling_duration = 0.0
        self._remodeling_cycles = 0
        if self.comm.Get_rank() == 0:
            print '[StressDrivenFibrosis] event', self.event_number
            print '  remodeling dt:', self.data['fib_remodeling_timestep']
            print '  phi_c min/mean/max:', \
                self.data['fib_phi_c_min'], \
                self.data['fib_phi_c_mean'], \
                self.data['fib_phi_c_max']
            print '  reference-volume residual:', \
                self.data['fib_reference_volume_residual']
            print '  total-mass residual:', \
                self.data['fib_total_mass_residual']
