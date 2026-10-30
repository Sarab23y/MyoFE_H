# -*- coding: utf-8 -*-
"""Pure configuration and update functions for volume-neutral fibrosis."""
import math


def _first(value, path):
    if not isinstance(value, list) or not value:
        raise ValueError('%s must be a non-empty JSON list' % path)
    return value[0]


def validate_fibrosis_config(config):
    enabled = bool(_first(config.get('enabled', [False]), 'fibrosis.enabled'))
    if not enabled:
        return {'enabled': False}
    model = _first(config.get('model', []), 'fibrosis.model')
    stimulus = _first(config.get('stimulus', []), 'fibrosis.stimulus')
    if model != 'stress_driven_constant_volume':
        raise ValueError('fibrosis.model must be stress_driven_constant_volume')
    supported_stimuli = (
        'cycle_averaged_total_fiber_stress',
        'cycle_averaged_total_fiber_stress_range')
    if stimulus not in supported_stimuli:
        raise ValueError(
            'fibrosis.stimulus is not supported')
    tau_c = float(_first(config.get('tau_c', []), 'fibrosis.tau_c'))
    phi_c_max = float(_first(
        config.get('phi_c_max', []), 'fibrosis.phi_c_max'))
    stress_epsilon = float(_first(
        config.get('stress_epsilon', []), 'fibrosis.stress_epsilon'))
    gamma_c = float(_first(config.get('gamma_c', [1.0]),
                           'fibrosis.gamma_c'))
    density = float(_first(config.get('density', [1.0]), 'fibrosis.density'))
    homeostatic_cycles = int(_first(
        config.get('homeostatic_cycles', [3]),
        'fibrosis.homeostatic_cycles'))
    remodeling_frequency_cycles = int(_first(
        config.get('remodeling_frequency_cycles', [4]),
        'fibrosis.remodeling_frequency_cycles'))
    activation_time = float(_first(
        config.get('activation_time', [0.0]), 'fibrosis.activation_time'))
    if any(math.isnan(value) or math.isinf(value)
           for value in (tau_c, stress_epsilon, density, phi_c_max,
                         activation_time)):
        raise ValueError('fibrosis numeric parameters must be finite')
    if tau_c <= 0.0 or stress_epsilon <= 0.0 or density <= 0.0:
        raise ValueError(
            'fibrosis tau_c, stress_epsilon, and density must be positive')
    if gamma_c <= 0.0 or math.isnan(gamma_c) or math.isinf(gamma_c):
        raise ValueError('fibrosis.gamma_c must be positive and finite')
    if phi_c_max <= 0.0 or phi_c_max >= 1.0:
        raise ValueError('fibrosis.phi_c_max must be between zero and one')
    if homeostatic_cycles < 1 or remodeling_frequency_cycles < 1 or \
            activation_time < 0.0:
        raise ValueError('fibrosis cycle counts must be positive integers')
    flags = {'geometric_growth_enabled': False,
             'volume_preserving': True,
             'degradation_enabled': False}
    for key, expected in flags.items():
        actual = bool(_first(config.get(key, [expected]), 'fibrosis.' + key))
        if actual != expected:
            raise ValueError('fibrosis.%s must be %s for this model' %
                             (key, str(expected).lower()))
    result = {'enabled': True, 'model': model, 'stimulus': stimulus,
            'tau_c': tau_c, 'phi_c_max': phi_c_max,
            'stress_epsilon': stress_epsilon, 'gamma_c': gamma_c,
            'density': density,
            'activation_time': activation_time,
            'homeostatic_cycles': homeostatic_cycles,
            'remodeling_frequency_cycles': remodeling_frequency_cycles}
    if stimulus == 'cycle_averaged_total_fiber_stress_range':
        calibration = config.get('stress_reference_calibration', {})
        calibration_enabled = bool(_first(
            calibration.get('enabled', [False]),
            'fibrosis.stress_reference_calibration.enabled'))
        if calibration_enabled:
            method = _first(calibration.get('method', []),
                            'fibrosis.stress_reference_calibration.method')
            cycles = int(_first(calibration.get('cycles', []),
                                'fibrosis.stress_reference_calibration.cycles'))
            lower = float(_first(
                calibration.get('lower_percentile', []),
                'fibrosis.stress_reference_calibration.lower_percentile'))
            upper = float(_first(
                calibration.get('upper_percentile', []),
                'fibrosis.stress_reference_calibration.upper_percentile'))
            if method != 'global_spatial_percentile':
                raise ValueError('Unsupported stress-reference calibration')
            if cycles < 1:
                raise ValueError('Calibration cycles must be positive')
            if any(math.isnan(value) or math.isinf(value)
                   for value in (lower, upper)) or not (
                    0.0 <= lower < upper <= 100.0):
                raise ValueError(
                    'Calibration percentiles must satisfy '
                    '0 <= lower < upper <= 100')
            result['stress_reference_calibration'] = {
                'enabled': True, 'method': method, 'cycles': cycles,
                'lower_percentile': lower, 'upper_percentile': upper}
        else:
            try:
                stress_min = float(_first(
                    config.get('S_ff_min', []), 'fibrosis.S_ff_min'))
                stress_max = float(_first(
                    config.get('S_ff_max', []), 'fibrosis.S_ff_max'))
            except (TypeError, ValueError):
                raise ValueError(
                    'fibrosis.S_ff_min and S_ff_max require measured numbers')
            if any(math.isnan(value) or math.isinf(value)
                   for value in (stress_min, stress_max)):
                raise ValueError('fibrosis stress references must be finite')
            if stress_max <= stress_min:
                raise ValueError('fibrosis.S_ff_max must exceed fibrosis.S_ff_min')
            result['S_ff_min'] = stress_min
            result['S_ff_max'] = stress_max
            result['stress_reference_calibration'] = {'enabled': False}
    return result


def effective_fibrosis_stimulus(stimulus, gamma_c):
    """Return ``max(G_c, 0)**gamma_c`` without clipping large values."""
    gamma_c = float(gamma_c)
    if gamma_c <= 0.0 or math.isnan(gamma_c) or math.isinf(gamma_c):
        raise ValueError('fibrosis.gamma_c must be positive and finite')
    effective = []
    for raw_value in stimulus:
        raw_value = max(float(raw_value), 0.0)
        if math.isnan(raw_value) or math.isinf(raw_value):
            raise ValueError('Fibrosis stimulus must be finite')
        value = raw_value**gamma_c
        if math.isnan(value) or math.isinf(value):
            raise ValueError('Effective fibrosis stimulus is not finite')
        effective.append(value)
    return effective

def calibrated_stress_references(values, lower_percentile, upper_percentile):
    """Return linear-interpolated percentiles and observed extrema.

    This operates on the complete global sample assembled by the caller. It
    deliberately reports the observed extrema separately so percentile-based
    outlier exclusion remains visible.
    """
    finite_values = []
    for value in values:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            raise ValueError('Baseline fiber stress must be finite')
        finite_values.append(value)
    if not finite_values:
        raise ValueError('Baseline fiber stress sample is empty')
    lower_percentile = float(lower_percentile)
    upper_percentile = float(upper_percentile)
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError('Stress percentiles are invalid')
    finite_values.sort()

    def percentile(percent):
        position = (len(finite_values) - 1)*percent/100.0
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        if lower_index == upper_index:
            return finite_values[lower_index]
        fraction = position - lower_index
        return (finite_values[lower_index]*(1.0-fraction) +
                finite_values[upper_index]*fraction)

    lower = percentile(lower_percentile)
    upper = percentile(upper_percentile)
    if upper <= lower:
        raise ValueError('Calibrated stress-reference range is degenerate')
    return lower, upper, finite_values[0], finite_values[-1]

def stress_range_signal(stress, stress_min, stress_max, gamma_c, tau_c):
    """Return bounded range coordinate and rate-valued fibrosis signal.

    ``r = clip((stress - stress_min)/(stress_max - stress_min), 0, 1)``
    and ``G_c = r**gamma_c/tau_c``.
    """
    stress_min = float(stress_min)
    stress_max = float(stress_max)
    tau_c = float(tau_c)
    if any(math.isnan(value) or math.isinf(value)
           for value in (stress_min, stress_max, tau_c)):
        raise ValueError('Stress limits and tau_c must be finite')
    if stress_max <= stress_min:
        raise ValueError('S_ff_max must exceed S_ff_min')
    if tau_c <= 0.0:
        raise ValueError('Fibrosis tau_c must be positive')
    normalized = []
    for value in stress:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            raise ValueError('Cycle-averaged fiber stress must be finite')
        ratio = (value - stress_min)/(stress_max - stress_min)
        normalized.append(min(max(ratio, 0.0), 1.0))
    effective = effective_fibrosis_stimulus(normalized, gamma_c)
    signal = [value/tau_c for value in effective]
    return normalized, signal


def exact_collagen_update_from_signal(phi_c, signal, dt, phi_c_max):
    """Exact bounded update when ``signal`` already has inverse-time units."""
    if dt < 0.0:
        raise ValueError('Fibrosis dt must be nonnegative')
    if len(phi_c) != len(signal):
        raise ValueError('Collagen fraction and signal sizes differ')
    updated = []
    for old_value, signal_value in zip(phi_c, signal):
        old_value = float(old_value)
        signal_value = float(signal_value)
        if math.isnan(signal_value) or math.isinf(signal_value) or \
                signal_value < 0.0:
            raise ValueError('Fibrosis signal must be finite and nonnegative')
        if old_value < 0.0 or old_value > phi_c_max + 1.0e-12:
            raise ValueError('Current collagen fraction is outside its bounds')
        value = old_value + (phi_c_max - old_value)*(
            1.0 - math.exp(-signal_value*float(dt)))
        updated.append(min(max(value, old_value), phi_c_max))
    return updated


def exact_collagen_update(phi_c, stimulus, dt, tau_c, phi_c_max,
                          gamma_c=1.0):
    """Exact bounded update for constant nonnegative stimulus over ``dt``."""
    if dt < 0.0 or tau_c <= 0.0:
        raise ValueError('Fibrosis dt must be nonnegative and tau_c positive')
    if len(phi_c) != len(stimulus):
        raise ValueError('Collagen fraction and stimulus sizes differ')
    effective = effective_fibrosis_stimulus(stimulus, gamma_c)
    updated = []
    for old_value, stimulus_value in zip(phi_c, effective):
        old_value = float(old_value)
        stimulus_value = max(float(stimulus_value), 0.0)
        if old_value < 0.0 or old_value > phi_c_max + 1.0e-12:
            raise ValueError('Current collagen fraction is outside its bounds')
        value = old_value + (phi_c_max - old_value)*(
            1.0 - math.exp(-stimulus_value*float(dt)/float(tau_c)))
        updated.append(min(max(value, old_value), phi_c_max))
    return updated


def validate_fraction_compatibility(phi_m, phi_g, phi_c, phi_c_max):
    """Validate a collagen cap against configured local mixture fractions."""
    if not (len(phi_m) == len(phi_g) == len(phi_c)):
        raise ValueError('Configured constituent fraction sizes differ')
    for point, values in enumerate(zip(phi_m, phi_g, phi_c)):
        myofiber, ground, collagen = [float(value) for value in values]
        if min(myofiber, ground, collagen) < 0.0 or \
                abs(myofiber + ground + collagen - 1.0) > 1.0e-8:
            raise ValueError(
                'Configured fractions are inadmissible at point %d' % point)
        if phi_c_max < collagen or phi_c_max > collagen + ground:
            raise ValueError(
                'fibrosis.phi_c_max is incompatible with configured '
                'fractions at point %d' % point)
