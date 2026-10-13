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
    if stimulus != 'cycle_averaged_total_fiber_stress':
        raise ValueError(
            'fibrosis.stimulus must be cycle_averaged_total_fiber_stress')
    tau_c = float(_first(config.get('tau_c', []), 'fibrosis.tau_c'))
    phi_c_max = float(_first(
        config.get('phi_c_max', []), 'fibrosis.phi_c_max'))
    stress_epsilon = float(_first(
        config.get('stress_epsilon', []), 'fibrosis.stress_epsilon'))
    density = float(_first(config.get('density', [1.0]), 'fibrosis.density'))
    homeostatic_cycles = int(_first(
        config.get('homeostatic_cycles', [3]),
        'fibrosis.homeostatic_cycles'))
    remodeling_frequency_cycles = int(_first(
        config.get('remodeling_frequency_cycles', [4]),
        'fibrosis.remodeling_frequency_cycles'))
    activation_time = float(_first(
        config.get('activation_time', [0.0]), 'fibrosis.activation_time'))
    if tau_c <= 0.0 or stress_epsilon <= 0.0 or density <= 0.0:
        raise ValueError(
            'fibrosis tau_c, stress_epsilon, and density must be positive')
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
    return {'enabled': True, 'model': model, 'stimulus': stimulus,
            'tau_c': tau_c, 'phi_c_max': phi_c_max,
            'stress_epsilon': stress_epsilon, 'density': density,
            'activation_time': activation_time,
            'homeostatic_cycles': homeostatic_cycles,
            'remodeling_frequency_cycles': remodeling_frequency_cycles}


def exact_collagen_update(phi_c, stimulus, dt, tau_c, phi_c_max):
    """Exact bounded update for constant nonnegative stimulus over ``dt``."""
    if dt < 0.0 or tau_c <= 0.0:
        raise ValueError('Fibrosis dt must be nonnegative and tau_c positive')
    if len(phi_c) != len(stimulus):
        raise ValueError('Collagen fraction and stimulus sizes differ')
    updated = []
    for old_value, stimulus_value in zip(phi_c, stimulus):
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
