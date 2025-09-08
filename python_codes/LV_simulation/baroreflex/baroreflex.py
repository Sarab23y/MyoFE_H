# -*- coding: utf-8 -*-
"""
Created on Sep 8 2025

@author: SARA
"""
import numpy as np
import json
from scipy.integrate import odeint

class baroreflex:
    """Baroreflex with pressure filtering, deadbanded error, and smoothed/clipped actuation."""

    def __init__(self, baro_structure, parent_circulation, pressure=0):
        # Parent
        self.parent_circulation = parent_circulation

        # -------- Core model params (from structure) --------
        self.model = dict()
        self.model['baro_b_slope'] = baro_structure['b_slope'][0]
        self.model['baro_k_drive'] = baro_structure['k_drive'][0]
        self.model['baro_k_recov'] = baro_structure['k_recov'][0]

        # New: pressure LPF time constant (s) and deadband in b-units
        self.model['tau_p'] = baro_structure.get('tau_p', [1.5])[0]       # ~1–3 beats
        self.model['deadband_b'] = baro_structure.get('deadband_b', [0.015])[0]  # ~±1–2%

        # -------- Data (state) --------
        self.data = dict()
        self.data['baro_b_setpoint'] = baro_structure['b_setpoint'][0]
        self.data['P_filt'] = float(pressure)                              # filtered pressure state
        self.data['baro_b'] = self.return_b(self.data['P_filt'])
        self.data['baro_c'] = 0.5                                          # control signal state (0..1)

        # -------- Controls wiring --------
        self.controls = []
        if ('controls' in baro_structure):
            baro_cont = baro_structure['controls']['control']
            for bc in baro_cont:
                if bc['level'][0] in ['myofilaments', 'membranes']:
                    for i, h in enumerate(self.parent_circulation.hs_objs_list):
                        self.controls.append(reflex_control(bc, self.parent_circulation, index=i))
                else:
                    self.controls.append(reflex_control(bc, self.parent_circulation))

        # expose rc values in data
        for bc in self.controls:
            if bc.data['level'] in ['myofilaments', 'membranes']:
                k = bc.data['level'] + '_' + bc.data['variable'] + '_rc_' + str(bc.data['index'])
            else:
                k = bc.data['level'] + '_' + bc.data['variable'] + '_rc'
            self.data[k] = bc.data['rc']

    # ---------------- Core step ----------------
    def implement_time_step(self, pressure, time_step, reflex_active=0):
        """Advance baroreflex by one time step (seconds)."""

        # 1) Filter the sensed pressure (baroreceptors don't react to beat-to-beat spikes)
        solP = odeint(self.diff_Pf, self.data['P_filt'], [0, time_step], args=((pressure,),))
        self.data['P_filt'] = float(solP[-1].item())

        # 2) Update sigmoidal afferent b using filtered pressure
        self.data['baro_b'] = self.return_b(self.data['P_filt'])

        # 3) Update central control signal c (with deadbanded error); clip to [0,1]
        sol = odeint(self.diff_c, self.data['baro_c'], [0, time_step], args=((reflex_active,),))
        self.data['baro_c'] = float(np.clip(sol[-1].item(), 0.0, 1.0))

        # 4) Push control outputs to the plant with smoothing & fractional caps
        for bc in self.controls:
            bc.implement_time_step(time_step, self.data['baro_c'], reflex_active)
            y_target = bc.return_output()

            # helper: smooth + cap fractional change
            def _smooth_cap(cur, target, alpha, max_frac):
                y_s = cur + alpha * (target - cur)          # exponential smoothing
                if cur != 0.0:
                    lo = cur * (1.0 - max_frac)
                    hi = cur * (1.0 + max_frac)
                    y_s = min(max(y_s, lo), hi)              # cap per-step fraction
                return y_s

            # write-throughs per level
            if (bc.data['level'] == 'heart_rate'):
                # heart rate can only update once per cycle
                if (self.parent_circulation.hr.data['t_RR'] ==
                    (self.parent_circulation.hr.data['t_active_period'] +
                     self.parent_circulation.hr.data['t_quiescent_period'])):
                    cur = self.parent_circulation.hr.data[bc.data['variable']]
                    alpha = bc.data.get('alpha', 0.25)
                    maxf  = bc.data.get('max_frac', 0.02)
                    self.parent_circulation.hr.data[bc.data['variable']] = _smooth_cap(cur, y_target, alpha, maxf)

            if (bc.data['level'] == 'membranes'):
                if not (bc.data['index'] in self.parent_circulation.infarct_regions or
                        bc.data['index'] in self.parent_circulation.border_zone_regions):
                    h = self.parent_circulation.hs_objs_list[bc.data['index']]
                    cur = h.memb.data[bc.data['variable']]
                    alpha = bc.data.get('alpha', 0.25)
                    maxf  = bc.data.get('max_frac', 0.02)
                    h.memb.data[bc.data['variable']] = _smooth_cap(cur, y_target, alpha, maxf)

            if (bc.data['level'] == 'myofilaments'):
                if not (bc.data['index'] in self.parent_circulation.infarct_regions or
                        bc.data['index'] in self.parent_circulation.border_zone_regions):
                    h = self.parent_circulation.hs_objs_list[bc.data['index']]
                    cur = h.myof.data[bc.data['variable']]
                    alpha = bc.data.get('alpha', 0.25)
                    maxf  = bc.data.get('max_frac', 0.02)
                    h.myof.data[bc.data['variable']] = _smooth_cap(cur, y_target, alpha, maxf)

            if (bc.data['level'] == 'circulation'):
                cur = self.parent_circulation.circ.data[bc.data['variable']]
                alpha = bc.data.get('alpha', 0.25)
                maxf  = bc.data.get('max_frac', 0.02)
                self.parent_circulation.circ.data[bc.data['variable']] = _smooth_cap(cur, y_target, alpha, maxf)

            # expose rc values in data dict (for logging)
            if bc.data['level'] in ['myofilaments', 'membranes']:
                k = bc.data['level'] + '_' + bc.data['variable'] + '_rc_' + str(bc.data['index'])
            else:
                k = bc.data['level'] + '_' + bc.data['variable'] + '_rc'
            self.data[k] = bc.data['rc']

    # ---------------- Dynamics ----------------
    def return_b(self, pressure):
        """Baroreceptor afferent (0..1) from pressure via sigmoid."""
        return 1.0 / (1.0 + np.exp(-self.model['baro_b_slope'] *
                                   (pressure - self.data['baro_b_setpoint'])))

    def diff_Pf(self, Pf, t, P):
        """1st-order low-pass for sensed pressure."""
        return (P - Pf) / self.model['tau_p']

    def diff_c(self, c, t, reflex_active=False):
        """
        Rate of change of control signal c (0..1).
        Deadband small errors around b=0.5 and re-scale outside the deadband.
        """
        # deadbanded error on b
        e = self.data['baro_b'] - 0.5
        db = self.model['deadband_b']
        if abs(e) <= db:
            e_eff = 0.0
        else:
            e_eff = np.sign(e) * (abs(e) - db) / (0.5 - db)

        if reflex_active:
            if e_eff >= 0:   # pressure above setpoint
                dcdt = -self.model['baro_k_drive'] * e_eff * c
            else:            # pressure below setpoint
                dcdt = -self.model['baro_k_drive'] * e_eff * (1.0 - c)
        else:
            dcdt = -self.model['baro_k_recov'] * (c - 0.5)
        return dcdt


class reflex_control:
    """Single efferent control with smoothed/clipped response."""

    def __init__(self, control_struct, parent_circulation, index=0):
        self.data = dict()
        for k in list(control_struct.keys()):
            self.data[k] = control_struct[k][0]
        self.data['basal_value'] = 0.0
        self.data['rc'] = 0.5  # 0..1

        # Optional knobs (can be overridden in control_struct)
        self.data.setdefault('alpha', 0.25)     # write-through smoothing (0<α≤1)
        self.data.setdefault('max_frac', 0.02)  # cap per-step fractional change (2%)

        # Wire to plant to get basal values
        if (self.data['level'] == 'heart_rate'):
            self.data['basal_value'] = parent_circulation.hr.data[self.data['variable']]
        if (self.data['level'] == 'membranes'):
            self.data['index'] = index
            self.data['basal_value'] = parent_circulation.hs_objs_list[index].memb.data[self.data['variable']]
        if (self.data['level'] == 'myofilaments'):
            self.data['index'] = index
            self.data['basal_value'] = parent_circulation.hs_objs_list[index].myof.data[self.data['variable']]
        if (self.data['level'] == 'circulation'):
            self.data['basal_value'] = parent_circulation.circ.data[self.data['variable']]

        # Values at max parasympathetic/sympathetic drive
        self.data['para_value'] = self.data['para_factor'] * self.data['basal_value']
        self.data['symp_value'] = self.data['symp_factor'] * self.data['basal_value']

    def implement_time_step(self, time_step, c, reflex_active=0):
        # integrate rc, clip to [0,1]
        sol = odeint(self.diff_rc, self.data['rc'], [0, time_step], args=((c, reflex_active),))
        self.data['rc'] = float(np.clip(sol[-1].item(), 0.0, 1.0))

    def diff_rc(self, y, t, c, reflex_active=0):
        # Recovery/drive for efferent rc
        if reflex_active:
            if c > 0.5:
                drcdt = self.data['k_drive'] * ((c - 0.5) / 0.5) * (1.0 - y)
            else:
                drcdt = self.data['k_drive'] * ((c - 0.5) / 0.5) * y
        else:
            drcdt = -1.0 * self.data['k_recov'] * (y - 0.5)
        return drcdt

    def return_output(self):
        """Map rc∈[0,1] to physical variable between para_value and symp_value."""
        rc = self.data['rc']
        if rc >= 0.5:
            m = (self.data['symp_value'] - self.data['basal_value']) / 0.5
            y = self.data['basal_value'] + m * (rc - 0.5)
        else:
            m = (self.data['basal_value'] - self.data['para_value']) / 0.5
            y = self.data['basal_value'] + m * (rc - 0.5)
        return y
