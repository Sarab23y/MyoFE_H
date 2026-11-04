"""Optional, non-invasive local mechanical-health diagnostics.

The helpers in this module are dependency-light and Python-2.7 compatible.
The runtime class imports DOLFIN lazily so the numerical rules can be tested
without the legacy FEniCS image.
"""
from __future__ import division, print_function
import json
import os
import numpy as np


def kinematic_residuals(stretch, strain, hsl, hsl0):
    """Return diagnostic residuals for E_ff and hsl kinematic identities."""
    stretch = np.asarray(stretch, dtype=float)
    return (np.asarray(strain, dtype=float) - 0.5*(stretch*stretch - 1.0),
            np.asarray(hsl, dtype=float) - stretch*np.asarray(hsl0, dtype=float))


def anomaly_masks(fields, thresholds):
    """Return named warning masks; never clip or modify supplied fields."""
    masks = {}
    for name, values in fields.items():
        masks['nonfinite_' + name] = ~np.isfinite(np.asarray(values))
    j = np.asarray(fields['J'])
    masks['nonpositive_J'] = np.isfinite(j) & (j <= 0.0)
    masks['small_positive_J'] = np.isfinite(j) & (j > 0.0) & \
        (j < float(thresholds['minimum_positive_J']))
    stretch = np.asarray(fields['fiber_stretch_raw'])
    masks['extreme_shortening'] = np.isfinite(stretch) & \
        (stretch < float(thresholds['minimum_fiber_stretch']))
    hsl = np.asarray(fields['hsl'])
    masks['short_hsl'] = np.isfinite(hsl) & \
        (hsl < float(thresholds['minimum_hsl_nm']))
    stress_limit = float(thresholds['maximum_absolute_stress_pa'])
    for name in thresholds.get('stress_fields', ()):
        values = np.asarray(fields[name])
        masks['extreme_' + name] = np.isfinite(values) & \
            (np.abs(values) > stress_limit)
    if 'fiber_strain_identity_error' in fields:
        values = np.asarray(fields['fiber_strain_identity_error'])
        masks['kinematic_fiber_strain_mismatch'] = np.isfinite(values) & \
            (np.abs(values) > float(thresholds['kinematic_relative_tolerance']))
    if 'hsl_identity_error_nm' in fields:
        values = np.asarray(fields['hsl_identity_error_nm'])
        scale = np.maximum(np.abs(np.asarray(fields['hsl'])), 1.0)
        masks['kinematic_hsl_mismatch'] = np.isfinite(values) & \
            (np.abs(values)/scale > float(thresholds['kinematic_relative_tolerance']))
    return masks


class MechanicalHealthDiagnostic(object):
    """Sample selected quadrature fields and write only anomalous points."""
    def __init__(self, config, parent):
        self.parent = parent
        self.mesh = parent.mesh
        self.comm = parent.comm
        self.frequency = int(config.get('check_frequency_n', [100])[0])
        self.maximum_records = int(config.get('maximum_records_per_check', [8])[0])
        self.tracked_dofs = set(int(value) for value in
            config.get('tracked_global_quadrature_dofs', [[]])[0])
        self.thresholds = {
            'minimum_positive_J': float(config.get('minimum_positive_J', [0.2])[0]),
            'minimum_fiber_stretch': float(config.get('minimum_fiber_stretch', [0.5])[0]),
            'minimum_hsl_nm': float(config.get('minimum_hsl_nm', [500.0])[0]),
            'maximum_absolute_stress_pa': float(
                config.get('maximum_absolute_stress_pa', [1.0e9])[0]),
            'kinematic_relative_tolerance': float(
                config.get('kinematic_relative_tolerance', [1.0e-8])[0]),
            'stress_fields': ('incompressibility_stress', 'xi_passive_stress',
                              'ground_matrix_stress', 'collagen_stress',
                              'active_stress', 'total_passive_stress',
                              'total_fiber_stress')}
        output = parent.instruction_data['output_handler']['output_data_path'][0]
        self.path = os.path.join(os.path.dirname(output),
                                 'mechanical_health_events.jsonl')
        if self.comm.Get_rank() == 0:
            directory = os.path.dirname(self.path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(self.path, 'w') as handle:
                handle.write(json.dumps({'type': 'metadata',
                    'stress_units': 'Pa', 'hsl_units': 'nm',
                    'stretch_and_J_units': 'dimensionless',
                    'thresholds': self.thresholds,
                    'warning': 'Diagnostic only; no values are clipped and the solver is not stopped.'}) + '\n')
        self.step = 0
        self._material_point_metadata = self._build_material_point_metadata()

    def _build_material_point_metadata(self):
        """Map local Quadrature DOFs to cell and within-cell point IDs."""
        qspace = self.mesh.model['function_spaces']['quadrature_space']
        mesh = self.mesh.model['mesh']; dofmap = qspace.dofmap()
        metadata = {}
        try:
            global_cells = mesh.topology().global_indices(mesh.topology().dim())
        except Exception:
            global_cells = None
        for local_cell in range(mesh.num_cells()):
            for quadrature_point, local_dof in enumerate(
                    dofmap.cell_dofs(local_cell)):
                local_dof = int(local_dof)
                if local_dof not in metadata:
                    metadata[local_dof] = {
                        'local_cell': int(local_cell),
                        'global_cell': (int(global_cells[local_cell])
                                        if global_cells is not None else None),
                        'quadrature_point_in_cell': int(quadrature_point)}
        return metadata

    def _values(self):
        from .fenics_2017_compat import (fiber_stress_components,
            project_quadrature)
        f = self.mesh.model['functions']; qspace = self.mesh.model['function_spaces']['quadrature_space']
        # These fields have already been evaluated at the identical material
        # points by the normal mechanics timestep.  Reuse them instead of
        # asking FFC 2017 to compile duplicate diagnostic forms.
        values = {
            'fiber_stretch_raw': np.asarray(
                self.parent.data['fiber_stretch_raw']).copy(),
            'hsl': np.asarray(self.parent.data['hsl']).copy(),
            'hsl0': f['hsl0'].vector().get_local().copy()}
        expressions = {
            'fiber_strain': f['fiber_strain'], 'J': f['J']}
        for name, expression in expressions.items():
            values[name] = project_quadrature(
                expression, qspace, 'mechanical_health_' + name,
                self.comm).vector().get_local()
        values.update(fiber_stress_components(
            f, qspace, self.comm, 'mechanical_health'))
        return values

    def record_time_step(self, unused_time_step):
        self.step += 1
        if self.step % self.frequency:
            return
        fields = self._values()
        strain_error, hsl_error = kinematic_residuals(
            fields['fiber_stretch_raw'], fields['fiber_strain'],
            fields['hsl'], fields['hsl0'])
        fields['fiber_strain_identity_error'] = strain_error
        fields['hsl_identity_error_nm'] = hsl_error
        masks = anomaly_masks(fields, self.thresholds)
        qspace = self.mesh.model['function_spaces']['quadrature_space']
        dofmap = qspace.dofmap()
        coordinates = qspace.tabulate_dof_coordinates().reshape(
            (-1, self.mesh.model['mesh'].geometry().dim()))
        local = []
        for reason, mask in masks.items():
            indices = np.where(mask)[0]
            if len(indices) > self.maximum_records:
                if reason.startswith('extreme_'):
                    source_name = reason.replace('extreme_', '')
                    score = np.abs(fields[source_name][indices])
                else:
                    source_name = {'nonpositive_J': 'J',
                        'small_positive_J': 'J',
                        'extreme_shortening': 'fiber_stretch_raw',
                        'short_hsl': 'hsl',
                        'kinematic_fiber_strain_mismatch':
                            'fiber_strain_identity_error',
                        'kinematic_hsl_mismatch':
                            'hsl_identity_error_nm'}.get(
                            reason, reason.replace('nonfinite_', ''))
                    score = -np.nan_to_num(fields[source_name][indices])
                indices = indices[np.argsort(score)[-self.maximum_records:]]
            field = {'nonpositive_J': 'J', 'small_positive_J': 'J',
                     'extreme_shortening': 'fiber_stretch_raw',
                     'short_hsl': 'hsl',
                     'kinematic_fiber_strain_mismatch':
                         'fiber_strain_identity_error',
                     'kinematic_hsl_mismatch': 'hsl_identity_error_nm'}.get(
                         reason, reason.replace('nonfinite_', '').replace(
                             'extreme_', ''))
            for index in indices:
                gid = int(dofmap.local_to_global_index(int(index)))
                value = fields.get(field, fields.get('J'))[index]
                if not np.isfinite(value):
                    severity = None
                elif reason.startswith('extreme_') and \
                        reason != 'extreme_shortening':
                    severity = float(abs(value))
                elif reason in ('nonpositive_J', 'small_positive_J',
                                'extreme_shortening', 'short_hsl'):
                    severity = float(-value)
                else:
                    severity = float(abs(value))
                local.append({'time_s': float(self.parent.data['time']),
                    'reason': reason, 'field': field, 'value': float(value),
                    'global_quadrature_dof_current_partition': gid,
                    'reference_coordinate': [float(x) for x in coordinates[index]],
                    'mpi_rank': int(self.comm.Get_rank()),
                    'severity': severity,
                    'element': self._material_point_metadata.get(int(index), {})})
        for index in range(len(fields['J'])):
            gid = int(dofmap.local_to_global_index(index))
            if gid in self.tracked_dofs:
                local.append({'time_s': float(self.parent.data['time']),
                    'reason': 'tracked_current_partition_dof', 'field': 'fiber_stretch_raw',
                    'value': float(fields['fiber_stretch_raw'][index]),
                    'values': dict((name, float(values[index]))
                                   for name, values in fields.items()),
                    'global_quadrature_dof_current_partition': gid,
                    'reference_coordinate': [float(x) for x in coordinates[index]],
                    'mpi_rank': int(self.comm.Get_rank()),
                    'severity': 0.0,
                    'element': self._material_point_metadata.get(int(index), {})})
        gathered = self.comm.gather(local, root=0)
        if self.comm.Get_rank() == 0:
            records = [item for rank_items in gathered for item in rank_items]
            records.sort(key=lambda item: (item['reason'],
                -(item['severity'] if item['severity'] is not None else float('inf')),
                item['reference_coordinate'], item['mpi_rank']))
            limited = []
            for reason in sorted(set(item['reason'] for item in records)):
                limited.extend([item for item in records
                                if item['reason'] == reason][:self.maximum_records])
            records = limited
            if records:
                print('WARNING: mechanical-health diagnostic recorded %d '
                      'quadrature-point events at t=%.6g s; see %s' %
                      (len(records), float(self.parent.data['time']), self.path))
                with open(self.path, 'a') as handle:
                    for record in records:
                        handle.write(json.dumps(record, sort_keys=True) + '\n')
