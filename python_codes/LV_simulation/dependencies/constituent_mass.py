# -*- coding: utf-8 -*-
"""Referential constituent-density bookkeeping at material points."""


CONSTITUENTS = ("myofiber", "ground_matrix", "collagen")
FRACTION_KEYS = {
    "myofiber": "phi_m",
    "ground_matrix": "phi_g",
    "collagen": "phi_c",
}


def _first(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("Expected a non-empty configuration list")
        return value[0]
    return value


def validate_mass_bookkeeping_config(config):
    """Return a normalized configuration without mutating user input."""
    if config is None:
        return {"enabled": False}
    enabled = bool(_first(config.get("enabled", [False])))
    if not enabled:
        return {"enabled": False}

    density = float(_first(config.get("density", [1.0])))
    if density <= 0.0:
        raise ValueError("constituent_mass_bookkeeping.density must be positive")
    mass_balance_tolerance = float(_first(
        config.get("mass_balance_tolerance", [1.0e-6])))
    if mass_balance_tolerance <= 0.0:
        raise ValueError(
            "constituent_mass_bookkeeping.mass_balance_tolerance must be positive")

    allocation = config.get("positive_growth_allocation", {})
    mode = _first(allocation.get("mode", ["proportional"]))
    if mode not in ("proportional", "fixed"):
        raise ValueError(
            "positive_growth_allocation.mode must be proportional or fixed")
    weights = None
    if mode == "fixed":
        weights = {}
        for constituent in CONSTITUENTS:
            if constituent not in allocation:
                raise ValueError(
                    "fixed positive growth allocation requires %s" %
                    constituent)
            weights[constituent] = float(_first(allocation[constituent]))
            if weights[constituent] < 0.0:
                raise ValueError("growth allocation weights must be nonnegative")
        if abs(sum(weights.values()) - 1.0) > 1.0e-8:
            raise ValueError("fixed positive growth allocation must sum to 1")

    return {
        "enabled": True,
        "density": density,
        "mode": mode,
        "weights": weights,
        "mass_balance_tolerance": mass_balance_tolerance,
    }


class ConstituentMassBookkeeper(object):
    """Track density per current reference volume and derive fractions."""

    def __init__(self, fractions, point_count, config):
        self.config = validate_mass_bookkeeping_config(config)
        if not self.config.get("enabled", False):
            raise ValueError("Cannot initialize disabled mass bookkeeping")
        self.density = self.config["density"]
        self.referential_density = dict((name, []) for name in CONSTITUENTS)
        for point in range(point_count):
            point_sum = sum(float(fractions[FRACTION_KEYS[name]][point])
                            for name in CONSTITUENTS)
            if abs(point_sum - 1.0) > 1.0e-8:
                raise ValueError(
                    "Initial constituent fractions must sum to 1 at point %d" %
                    point)
            for name in CONSTITUENTS:
                fraction = float(fractions[FRACTION_KEYS[name]][point])
                if fraction < 0.0 or fraction > 1.0:
                    raise ValueError("Initial constituent fractions must be in [0, 1]")
                self.referential_density[name].append(self.density*fraction)

    def fractions(self):
        result = dict((key, []) for key in FRACTION_KEYS.values())
        point_count = len(self.referential_density[CONSTITUENTS[0]])
        for point in range(point_count):
            total = sum(self.referential_density[name][point]
                        for name in CONSTITUENTS)
            if total <= 0.0:
                raise ValueError("Total constituent mass must remain positive")
            for name in CONSTITUENTS:
                result[FRACTION_KEYS[name]].append(
                    self.referential_density[name][point]/total)
        return result

    def update_positive_growth(self, incremental_volume_ratios,
                               geometric_volume_ratios):
        """Deposit on the old reference, then transport density through ALE.

        ``incremental_volume_ratios`` is det(Fg). ``geometric_volume_ratios``
        is the Jacobian of the solved ALE map from the old to new reference.
        Dividing the post-deposition density by the ALE Jacobian preserves the
        physical mass represented by the moved integration volume.
        """
        point_count = len(self.referential_density[CONSTITUENTS[0]])
        if len(incremental_volume_ratios) != point_count:
            raise ValueError("Growth-volume ratio count does not match points")
        if len(geometric_volume_ratios) != point_count:
            raise ValueError("ALE-volume ratio count does not match points")
        fractions_before = self.fractions()
        added_referential_density = dict(
            (name, [0.0]*point_count) for name in CONSTITUENTS)
        added_volume_ratio = [0.0]*point_count

        for point, ratio_value in enumerate(incremental_volume_ratios):
            ratio = float(ratio_value)
            geometric_ratio = float(geometric_volume_ratios[point])
            if ratio <= 0.0:
                raise ValueError("det(Fg) must be positive")
            if geometric_ratio <= 0.0:
                raise ValueError("ALE geometric volume ratio must be positive")
            positive_growth = max(ratio - 1.0, 0.0)
            added_volume_ratio[point] = positive_growth
            if self.config["mode"] == "proportional":
                weights = dict(
                    (name, fractions_before[FRACTION_KEYS[name]][point])
                    for name in CONSTITUENTS)
            else:
                weights = self.config["weights"]
            for name in CONSTITUENTS:
                deposited_density = self.density*positive_growth*weights[name]
                old_density = self.referential_density[name][point]
                self.referential_density[name][point] = \
                    (old_density + deposited_density)/geometric_ratio
                added_referential_density[name][point] = deposited_density

            transported_mass_per_old_volume = sum(
                self.referential_density[name][point]*geometric_ratio
                for name in CONSTITUENTS)
            old_mass_per_old_volume = sum(
                (self.referential_density[name][point]*geometric_ratio -
                 added_referential_density[name][point])
                for name in CONSTITUENTS)
            expected_mass_per_old_volume = \
                old_mass_per_old_volume + self.density*positive_growth
            if abs(transported_mass_per_old_volume -
                   expected_mass_per_old_volume) > \
                    1.0e-10*max(1.0, abs(expected_mass_per_old_volume)):
                raise RuntimeError(
                    "Constituent density transport violates local mass balance")

        return {
            "fractions": self.fractions(),
            "added_referential_density": added_referential_density,
            "added_volume_ratio": added_volume_ratio,
            "growth_volume_ratio": list(incremental_volume_ratios),
            "geometric_volume_ratio": list(geometric_volume_ratios),
            "referential_density": dict(
                (name, list(values))
                for name, values in self.referential_density.items()),
        }

    def exchange_ground_matrix_for_collagen(self, collagen_fractions):
        """Exchange ground-matrix density for collagen at fixed total density.

        This volume-neutral operation does not represent growth deposition:
        myofiber and total referential density remain unchanged pointwise.
        """
        point_count = len(self.referential_density[CONSTITUENTS[0]])
        if len(collagen_fractions) != point_count:
            raise ValueError("Collagen-fraction count does not match points")
        before_total = []
        exchanged_density = []
        for point, target_value in enumerate(collagen_fractions):
            target = float(target_value)
            if target < 0.0 or target > 1.0:
                raise ValueError("Target collagen fraction must be in [0, 1]")
            total = sum(self.referential_density[name][point]
                        for name in CONSTITUENTS)
            before_total.append(total)
            current_collagen = self.referential_density['collagen'][point]
            target_collagen = target*total
            increment = target_collagen - current_collagen
            if increment < -1.0e-12*max(1.0, total):
                raise ValueError(
                    "Constant-volume fibrosis cannot reduce collagen")
            increment = max(increment, 0.0)
            ground = self.referential_density['ground_matrix'][point]
            if increment > ground + 1.0e-12*max(1.0, total):
                raise ValueError(
                    "Insufficient ground-matrix density for collagen exchange")
            self.referential_density['collagen'][point] = \
                current_collagen + increment
            self.referential_density['ground_matrix'][point] = \
                ground - increment
            exchanged_density.append(increment)

        for point, expected in enumerate(before_total):
            actual = sum(self.referential_density[name][point]
                         for name in CONSTITUENTS)
            if abs(actual - expected) > 1.0e-12*max(1.0, expected):
                raise RuntimeError(
                    "Fibrosis exchange changed total referential density")
        return {
            "fractions": self.fractions(),
            "exchanged_density": exchanged_density,
            "referential_density": dict(
                (name, list(values))
                for name, values in self.referential_density.items()),
        }
