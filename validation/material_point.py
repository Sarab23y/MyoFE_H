#!/usr/bin/env python
"""Homogeneous material-point evaluation using the production Forms class."""
from __future__ import print_function

import csv
import json
import os
import sys

import numpy as np
from dolfin import (Constant, Function, FunctionSpace, UnitCubeMesh,
                    VectorFunctionSpace, as_tensor, assemble, dx)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_CODES = os.path.join(ROOT, 'python_codes')
if PYTHON_CODES not in sys.path:
    sys.path.insert(0, PYTHON_CODES)

from LV_simulation.dependencies.fibrosis_config import (  # noqa: E402
    validate_xi_ground_collagen)
from LV_simulation.dependencies.forms import Forms  # noqa: E402


class PrescribedCForms(Forms):
    """Use production energies/stresses at a prescribed homogeneous C."""

    def __init__(self, params, Cmat):
        self.prescribed_C = Cmat
        Forms.__init__(self, params)

    def Cmat(self):
        return self.prescribed_C

    def J(self):
        # All currently supported validation paths are isochoric.
        return Constant(1.0)


def load_material_input(config_path):
    with open(config_path, 'r') as handle:
        config = json.load(handle)
    input_path = config['input_json']
    if not os.path.isabs(input_path):
        input_path = os.path.normpath(os.path.join(ROOT, input_path))
    with open(input_path, 'r') as handle:
        instruction = json.load(handle)
    passive = instruction['mesh']['forms_parameters'][
        'passive_law_parameters']
    resolved = validate_xi_ground_collagen(passive)
    if resolved is None:
        raise ValueError('Material validation requires Xi-ground-collagen')
    return config, input_path, passive, resolved


def biaxial_F(lambda_f, lambda_s):
    """Incompressible aligned fiber-sheet stretch."""
    return np.diag([lambda_f, lambda_s, 1.0/(lambda_f*lambda_s)])


def shear_F(gamma, direction):
    """Isochoric simple shear in local material coordinates."""
    F = np.eye(3)
    if direction == 'fiber_along_sheet':
        F[0, 1] = gamma
    elif direction == 'sheet_along_fiber':
        F[1, 0] = gamma
    elif direction == 'fiber_along_normal':
        F[0, 2] = gamma
    else:
        raise ValueError('Unsupported shear direction: %s' % direction)
    return F


def _assemble_scalar(expression, mesh, volume):
    return float(assemble(expression*dx(domain=mesh))/volume)


def _assemble_tensor(expression, mesh, volume):
    return np.array([[_assemble_scalar(expression[i, j], mesh, volume)
                      for j in range(3)] for i in range(3)])


def evaluate_state(F_numpy, passive_parameters):
    """Evaluate production energy and PK2 stress at one homogeneous state.

    Constraint pressure is chosen once from zero total Cauchy traction in the
    local sheet-normal direction. Constituent stresses remain energetic
    contributions; the pressure is reported as a separate constraint term.
    """
    mesh = UnitCubeMesh(1, 1, 1)
    displacement = Function(VectorFunctionSpace(mesh, 'CG', 1))
    pressure = Constant(0.0)
    C_numpy = np.dot(F_numpy.T, F_numpy)
    C = as_tensor(C_numpy.tolist())
    lambda_f = np.sqrt(C_numpy[0, 0])
    params = {
        'displacement_variable': displacement,
        'pressure_variable': pressure,
        'fiber': Constant((1.0, 0.0, 0.0)),
        'sheet': Constant((0.0, 1.0, 0.0)),
        'sheet-normal': Constant((0.0, 0.0, 1.0)),
        'hsl0': Constant(1.0),
        'incompressible': True,
        'ground_matrix': passive_parameters['ground_matrix'],
        'myofiber': passive_parameters['myofiber'],
        'collagen': passive_parameters['collagen'],
        'phi_m': passive_parameters['phi_m'],
        'phi_g': passive_parameters['phi_g'],
        'phi_c': passive_parameters['phi_c'],
    }
    forms = PrescribedCForms(params, C)
    volume = assemble(Constant(1.0)*dx(domain=mesh))
    energies = forms._passive_energy_components(C, Constant(lambda_f))
    phi = [passive_parameters['phi_g'][0],
           passive_parameters['phi_m'][0],
           passive_parameters['phi_c'][0]]
    weighted_energies = [phi[i]*_assemble_scalar(energies[i], mesh, volume)
                         for i in range(3)]

    components = forms._passive_stress_components(Constant(lambda_f))
    stresses = [_assemble_tensor(components[i], mesh, volume)
                for i in (1, 2, 3)]
    J = float(np.linalg.det(F_numpy))
    cauchy = [np.dot(np.dot(F_numpy, S), F_numpy.T)/J for S in stresses]
    material_cauchy = sum(cauchy)
    constraint_pressure = material_cauchy[2, 2]
    constraint_cauchy = -constraint_pressure*np.eye(3)
    total_cauchy = material_cauchy + constraint_cauchy
    C_inverse = np.linalg.inv(C_numpy)
    total_pk2 = sum(stresses) - constraint_pressure*J*C_inverse
    E = 0.5*(C_numpy - np.eye(3))

    names = ('ground', 'myofiber', 'collagen')
    result = {
        'lambda_f': float(lambda_f),
        'lambda_s': float(np.sqrt(C_numpy[1, 1])),
        'E_ff': float(E[0, 0]),
        'E_ss': float(E[1, 1]),
        'E_fs_tensor': float(E[0, 1]),
        'engineering_shear_gamma': float(F_numpy[0, 1] + F_numpy[1, 0]),
        'J': J,
        'constraint_pressure': float(constraint_pressure),
        'traction_normal_residual': float(total_cauchy[2, 2]),
        'energy_total': float(sum(weighted_energies)),
        'S_ff_total': float(total_pk2[0, 0]),
        'S_ss_total': float(total_pk2[1, 1]),
        'S_fs_total': float(total_pk2[0, 1]),
        'sigma_ff_total': float(total_cauchy[0, 0]),
        'sigma_ss_total': float(total_cauchy[1, 1]),
        'sigma_fs_total': float(total_cauchy[0, 1]),
    }
    for i, name in enumerate(names):
        result['energy_' + name] = float(weighted_energies[i])
        result['S_ff_' + name] = float(stresses[i][0, 0])
        result['S_ss_' + name] = float(stresses[i][1, 1])
        result['S_fs_' + name] = float(stresses[i][0, 1])
        result['sigma_ff_' + name] = float(cauchy[i][0, 0])
        result['sigma_ss_' + name] = float(cauchy[i][1, 1])
        result['sigma_fs_' + name] = float(cauchy[i][0, 1])
    return result


def write_results(rows, columns, output_csv, metadata):
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(output_csv, 'wb') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    metadata_path = os.path.splitext(output_csv)[0] + '_metadata.json'
    with open(metadata_path, 'w') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return metadata_path
