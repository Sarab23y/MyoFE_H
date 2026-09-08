# -*- coding: utf-8 -*-
"""Validation helpers for the Xi-ground-collagen passive-law input."""


PASSIVE_LAW = "Xi-ground-collagen"
REQUIRED_CONSTITUENT_PARAMETERS = (
    ("ground_matrix", "a_g"),
    ("ground_matrix", "b_g"),
    ("myofiber", "c2"),
    ("myofiber", "c3"),
    ("collagen", "a_cf"),
    ("collagen", "b_cf"),
    ("collagen", "a_cs"),
    ("collagen", "b_cs"),
    ("collagen", "a_cn"),
    ("collagen", "b_cn"),
)
REQUIRED_FRACTIONS = ("phi_m", "phi_g", "phi_c")


def _required_scalar(container, key, path):
    if key not in container:
        raise ValueError(
            "%s requires passive_law_parameters.%s" % (PASSIVE_LAW, path))
    value = container[key]
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(
            "%s requires non-empty list passive_law_parameters.%s" %
            (PASSIVE_LAW, path))
    if not isinstance(value[0], (int, float)):
        raise ValueError(
            "%s requires numeric passive_law_parameters.%s[0]" %
            (PASSIVE_LAW, path))
    return value[0]


def validate_xi_ground_collagen(passive_parameters, tolerance=1.0e-8):
    """Validate without replacing or normalizing any user-supplied value."""
    if ("passive_law" not in passive_parameters or
            not isinstance(passive_parameters["passive_law"], list) or
            len(passive_parameters["passive_law"]) == 0):
        raise ValueError("passive_law_parameters.passive_law is required")
    law = passive_parameters["passive_law"][0]
    if law != PASSIVE_LAW:
        return None

    resolved = {"passive_law": law}
    for group, key in REQUIRED_CONSTITUENT_PARAMETERS:
        if group not in passive_parameters or not isinstance(
                passive_parameters[group], dict):
            raise ValueError(
                "%s requires passive_law_parameters.%s.%s" %
                (PASSIVE_LAW, group, key))
        resolved[key] = _required_scalar(
            passive_parameters[group], key, "%s.%s" % (group, key))

    for key in REQUIRED_FRACTIONS:
        resolved[key] = _required_scalar(passive_parameters, key, key)
        if resolved[key] < 0.0 or resolved[key] > 1.0:
            raise ValueError(
                "%s requires passive_law_parameters.%s in [0, 1]" %
                (PASSIVE_LAW, key))

    fraction_sum = sum(resolved[key] for key in REQUIRED_FRACTIONS)
    if abs(fraction_sum - 1.0) > tolerance:
        raise ValueError(
            "%s requires phi_m + phi_g + phi_c = 1; got %.16g" %
            (PASSIVE_LAW, fraction_sum))
    resolved["fraction_sum"] = fraction_sum
    return resolved
