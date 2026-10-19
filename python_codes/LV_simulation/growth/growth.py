# -*- coding: utf-8 -*-
"""
Created on Thu July 31 18:58:51 2022

@author: Hossein
"""

import numpy as np
import json
from scipy.integrate import odeint
from dolfin import *


from mechanics import GrowthMechanicsClass
from ..dependencies.nsolver import NSolver
from ..dependencies.constituent_mass import (
    ConstituentMassBookkeeper, validate_mass_bookkeeping_config)

from scipy.integrate import odeint

class growth():

    def __init__(self, growth_structure,
                parent_circulation):
            
        # Set the parent circulation
        self.parent_circulation = parent_circulation
        self.mesh  = self.parent_circulation.mesh

        # Initialise the model dict
        self.model = dict()
        # Initialise the data dict
        self.data = dict()

        self.comm = parent_circulation.comm
    
        # create object for the mechanics 
        predefined_functions = dict()
        predefined_functions['facetboundaries'] = self.mesh.model['functions']['facetboundaries']
        predefined_functions['hsl0'] = self.mesh.model['functions']['hsl0']
        predefined_functions['f0'] = self.mesh.model['functions']['f0']
        predefined_functions['s0'] = self.mesh.model['functions']['s0']
        predefined_functions['n0'] = self.mesh.model['functions']['n0']
        predefined_mesh = self.mesh.model['mesh']
        self.mechan = GrowthMechanicsClass(self.parent_circulation,
                                            predefined_mesh=predefined_mesh,
                                            predefined_functions=predefined_functions)
        
        # create object for the solver of the growth
        self.solver = NSolver(self.parent_circulation,self.mechan,self.comm)

        for dir in ['fiber','sheet','sheet_normal']:
            for param in ['local_theta','global_theta','setpoint','stimulus','deviation']:
                name = 'gr_' + param + '_'+  dir
                size =  len(self.mechan.model['functions']['theta_'+dir].vector().get_local()[:])
                if param not in ['local_theta','global_theta']:
                    self.data[name] = np.zeros(size)
                else:
                    self.data[name] = np.ones(size)

        self.components = []
        if 'components' in growth_structure:
            comp = growth_structure['components']
            for c in comp:     
                self.components.append(growth_component(c,self))

                # define some global parameters useful for plotting and 
                # applying perturbations 
                g_obj= self.components[-1]

                """for param in ['local_theta','global_theta','setpoint','stimulus','deviation']:
                    name = 'gr_' + param + '_'+  g_obj.data['type']
                    size =  len(self.mechan.model['functions']['theta_' +g_obj.data['type'] ].vector().get_local()[:])
                    if param not in ['local_theta','global_theta']:
                        self.data[name] = np.zeros(size)
                    else:
                        self.data[name] = np.ones(size)"""

        # handle data for visualization plots
        self.initial_gr_cycle_counter = 1
        self.initial_gr_cycles = 3
        # handle activations 
        self.data['gr_start_active'] = 0
        self.data['gr_active'] = 0
        self.data['gr_setpoint_active'] = 0
        self.data['gr_stimulus_active'] = 0

        self.growth_frequency_n = 0
        if 'growth_frequency_n' in growth_structure:
            self.growth_frequency_n = \
                int(growth_structure['growth_frequency_n'][0])
        # allow 1 cardiac cycle at the begining of growth to happen 
        # in order to store setpoint data
        self.growth_frequency_n_counter = self.growth_frequency_n - (self.initial_gr_cycles-1)
        self.prescribed_growth_increment = {}
        if 'prescribed_growth_increment' in growth_structure:
            self.prescribed_growth_increment = dict(
                (name, float(value[0])) for name, value in
                growth_structure['prescribed_growth_increment'].items())
            for name, value in self.prescribed_growth_increment.items():
                if name not in ('fiber', 'sheet', 'sheet_normal'):
                    raise ValueError(
                        'Unknown prescribed growth direction: %s' % name)
                if value <= 0.0:
                    raise ValueError(
                        'prescribed_growth_increment.%s must be positive' %
                        name)

        self.mass_bookkeeping_config = validate_mass_bookkeeping_config(
            growth_structure.get('constituent_mass_bookkeeping'))
        self.constituent_mass_bookkeeper = None
        self.material_point_keys = None
        self.growth_event_number = 0
        self.global_constituent_mass = None
        if self.mass_bookkeeping_config.get('enabled', False):
            self._initialize_constituent_mass_bookkeeping()
        

    def assign_setpoint(self):

        if self.comm.Get_rank() == 0:
            print 'assigning setpoint'
        for comp in self.components:
            #assign setpoint to be mean of setpoint tracker
            comp.data['setpoint'] = \
                np.mean(comp.data['setpoint_tracker'],axis=0)
            
            set_name = 'gr_setpoint' + '_' +  comp.data['type']
            self.data[set_name] = comp.data['setpoint']
            
            # reset setpoint_tracker 
            comp.data['setpoint_tracker'] = []

        return 

    def _passive_fraction_functions(self, mesh_object):
        passive = mesh_object.model['functions']['dolfin_functions'][
            'passive_params']
        return dict((name, passive[name][-1])
                    for name in ('phi_m', 'phi_g', 'phi_c'))

    def _initialize_constituent_mass_bookkeeping(self):
        fraction_functions = self._passive_fraction_functions(self.mesh)
        fractions = dict((name, function.vector().get_local().tolist())
                         for name, function in fraction_functions.items())
        point_count = len(fractions['phi_m'])
        if any(len(values) != point_count for values in fractions.values()):
            raise ValueError('Constituent fraction fields have different sizes')
        self.material_point_keys = self._quadrature_material_keys(self.mesh)
        if len(self.material_point_keys) != point_count:
            raise ValueError(
                'Material-point identity count does not match fraction DOFs')
        self.constituent_mass_bookkeeper = ConstituentMassBookkeeper(
            fractions, point_count, self.mass_bookkeeping_config)
        self._store_constituent_mass_state(
            {'fractions': self.constituent_mass_bookkeeper.fractions(),
             'referential_density':
                 self.constituent_mass_bookkeeper.referential_density,
             'added_volume_ratio': [0.0]*point_count,
             'growth_volume_ratio': [1.0]*point_count,
             'geometric_volume_ratio': [1.0]*point_count})
        self.apply_constituent_fractions(self.mesh)
        self.apply_constituent_fractions(self.mechan)
        self.global_constituent_mass = self._integrate_constituent_masses(
            self.mesh)
        initial_volume = float(assemble(
            1.0*dx(domain=self.mesh.model['mesh']),
            form_compiler_parameters={'representation': 'uflacs'}))
        self.data['gr_growth_event_number'] = 0
        self.data['gr_global_mass_myofiber'] = \
            self.global_constituent_mass['myofiber']
        self.data['gr_global_mass_ground_matrix'] = \
            self.global_constituent_mass['ground_matrix']
        self.data['gr_global_mass_collagen'] = \
            self.global_constituent_mass['collagen']
        self.data['gr_global_mass_total'] = \
            self.global_constituent_mass['total']
        self.data['gr_mass_balance_residual'] = 0.0
        self.data['gr_mass_balance_residual_myofiber'] = 0.0
        self.data['gr_mass_balance_residual_ground_matrix'] = 0.0
        self.data['gr_mass_balance_residual_collagen'] = 0.0
        self.data['gr_pre_growth_myocardial_volume'] = initial_volume
        self.data['gr_post_growth_myocardial_volume'] = initial_volume
        self.data['gr_prescribed_positive_growth_volume'] = 0.0

    def _quadrature_material_keys(self, mesh_object):
        """Return stable (global cell, cell-local quadrature point) keys."""
        mesh = mesh_object.model['mesh']
        space = mesh_object.model['function_spaces']['quadrature_space']
        dofmap = space.dofmap()
        start, end = dofmap.ownership_range()
        local_to_global = dofmap.tabulate_local_to_global_dofs()
        keys_by_owned_index = {}
        topological_dimension = mesh.topology().dim()
        mesh.init(topological_dimension)
        global_cells = mesh.topology().global_indices(topological_dimension)
        for cell in cells(mesh):
            global_cell = int(global_cells[cell.index()])
            for local_point, local_dof in enumerate(
                    dofmap.cell_dofs(cell.index())):
                global_dof = int(local_to_global[local_dof])
                if start <= global_dof < end:
                    owned_index = global_dof - start
                    key = (global_cell, int(local_point))
                    if owned_index in keys_by_owned_index and \
                            keys_by_owned_index[owned_index] != key:
                        raise RuntimeError(
                            'Quadrature DOF has multiple material-point keys')
                    keys_by_owned_index[owned_index] = key
        expected = end - start
        if sorted(keys_by_owned_index.keys()) != list(range(expected)):
            raise RuntimeError(
                'Could not identify every owned scalar Quadrature DOF')
        return [keys_by_owned_index[index] for index in range(expected)]

    def _values_in_canonical_order(self, mesh_object, values):
        target_keys = self._quadrature_material_keys(mesh_object)
        if len(values) != len(target_keys):
            raise ValueError('Material-point value count changed')
        source = dict(zip(target_keys, values))
        if set(source.keys()) != set(self.material_point_keys):
            raise RuntimeError(
                'Material-point identities changed across reconstruction')
        return [source[key] for key in self.material_point_keys]

    def _store_constituent_mass_state(self, state):
        fractions = state['fractions']
        self.data['gr_phi_m'] = np.asarray(fractions['phi_m'])
        self.data['gr_phi_g'] = np.asarray(fractions['phi_g'])
        self.data['gr_phi_c'] = np.asarray(fractions['phi_c'])
        densities = state['referential_density']
        self.data['gr_density_myofiber'] = np.asarray(
            densities['myofiber'])
        self.data['gr_density_ground_matrix'] = np.asarray(
            densities['ground_matrix'])
        self.data['gr_density_collagen'] = np.asarray(
            densities['collagen'])
        self.data['gr_density_total'] = \
            self.data['gr_density_myofiber'] + \
            self.data['gr_density_ground_matrix'] + \
            self.data['gr_density_collagen']
        self.data['gr_last_added_volume_ratio'] = np.asarray(
            state['added_volume_ratio'])
        if 'growth_volume_ratio' in state:
            self.data['gr_growth_volume_ratio'] = np.asarray(
                state['growth_volume_ratio'])
            self.data['gr_geometric_volume_ratio'] = np.asarray(
                state['geometric_volume_ratio'])
        if 'added_referential_density' in state:
            additions = state['added_referential_density']
            self.data['gr_last_added_density_myofiber'] = np.asarray(
                additions['myofiber'])
            self.data['gr_last_added_density_ground_matrix'] = np.asarray(
                additions['ground_matrix'])
            self.data['gr_last_added_density_collagen'] = np.asarray(
                additions['collagen'])
        else:
            zeros = np.zeros(len(self.data['gr_phi_m']))
            self.data['gr_last_added_density_myofiber'] = zeros.copy()
            self.data['gr_last_added_density_ground_matrix'] = zeros.copy()
            self.data['gr_last_added_density_collagen'] = zeros.copy()

    def incremental_growth_volume_ratios(self):
        """Return local det(Fg) in the passive quadrature-point ordering."""
        functions = self.mechan.model['functions']
        Jg = functions['theta_fiber']*functions['theta_sheet']*\
            functions['theta_sheet_normal']
        projected = project(
            Jg, self.mechan.model['function_spaces']['quadrature_space'],
            form_compiler_parameters={'representation': 'uflacs'})
        return self._values_in_canonical_order(
            self.mechan, projected.vector().get_local().tolist())

    def update_constituent_masses(self, incremental_volume_ratios,
                                  geometric_volume_ratios):
        """Book positive growth and recalculate all three local fractions."""
        if self.constituent_mass_bookkeeper is None:
            return
        state = self.constituent_mass_bookkeeper.update_positive_growth(
            incremental_volume_ratios, geometric_volume_ratios)
        self._store_constituent_mass_state(state)
        self.growth_event_number += 1
        local_summary = dict(
            (name, ((float(np.min(values)), float(np.max(values)))
                    if len(values) else (float('inf'), float('-inf'))))
            for name, values in state['fractions'].items())
        summaries = self.comm.gather(local_summary, root=0)
        if self.comm.Get_rank() == 0:
            print '[GrowthConstituentMass] updated strain-energy fractions'
            for name in ('phi_m', 'phi_g', 'phi_c'):
                print '  %s min/max: %g %g' % (
                    name,
                    min(summary[name][0] for summary in summaries),
                    max(summary[name][1] for summary in summaries))

    def apply_constituent_fractions(self, mesh_object):
        """Apply current mass fractions to the Functions used by Forms."""
        if self.constituent_mass_bookkeeper is None:
            return
        fraction_functions = self._passive_fraction_functions(mesh_object)
        fractions = self.constituent_mass_bookkeeper.fractions()
        for name, function in fraction_functions.items():
            target_keys = self._quadrature_material_keys(mesh_object)
            if set(target_keys) != set(self.material_point_keys):
                raise RuntimeError(
                    'Material-point identities changed across reconstruction')
            canonical_values = dict(zip(self.material_point_keys,
                                        fractions[name]))
            values = np.asarray([canonical_values[key]
                                 for key in target_keys])
            if len(function.vector().get_local()) != len(values):
                raise ValueError(
                    'Growth changed the constituent fraction DOF count')
            function.vector().set_local(values)
            function.vector().apply('insert')
            mesh_object.model['functions'][name] = function

        density_names = {
            'myofiber': 'constituent_density_myofiber',
            'ground_matrix': 'constituent_density_ground_matrix',
            'collagen': 'constituent_density_collagen',
        }
        target_keys = self._quadrature_material_keys(mesh_object)
        for constituent, function_name in density_names.items():
            density_function = Function(
                mesh_object.model['function_spaces']['quadrature_space'])
            canonical_density = dict(zip(
                self.material_point_keys,
                self.constituent_mass_bookkeeper.referential_density[
                    constituent]))
            density_function.vector().set_local(np.asarray(
                [canonical_density[key] for key in target_keys]))
            density_function.vector().apply('insert')
            mesh_object.model['functions'][function_name] = density_function
        total_density = Function(
            mesh_object.model['function_spaces']['quadrature_space'])
        total_density.vector().set_local(
            mesh_object.model['functions'][
                'constituent_density_myofiber'].vector().get_local() +
            mesh_object.model['functions'][
                'constituent_density_ground_matrix'].vector().get_local() +
            mesh_object.model['functions'][
                'constituent_density_collagen'].vector().get_local())
        total_density.vector().apply('insert')
        mesh_object.model['functions'][
            'constituent_density_total'] = total_density

        for data_name, function_name in (
                ('gr_growth_volume_ratio', 'growth_volume_ratio'),
                ('gr_geometric_volume_ratio', 'growth_geometric_volume_ratio')):
            if data_name in self.data:
                function = Function(
                    mesh_object.model['function_spaces']['quadrature_space'])
                canonical = dict(zip(self.material_point_keys,
                                     self.data[data_name]))
                function.vector().set_local(np.asarray(
                    [canonical[key] for key in target_keys]))
                function.vector().apply('insert')
                mesh_object.model['functions'][function_name] = function

    def order_spatial_mass_data_for_mesh(self, mesh_object):
        """Put diagnostic arrays in the target mesh's owned-DOF order."""
        target_keys = self._quadrature_material_keys(mesh_object)
        if set(target_keys) != set(self.material_point_keys):
            raise RuntimeError(
                'Material-point identities changed before spatial output')
        for name in ('gr_phi_m', 'gr_phi_g', 'gr_phi_c',
                     'gr_density_myofiber', 'gr_density_ground_matrix',
                     'gr_density_collagen', 'gr_density_total',
                     'gr_last_added_volume_ratio',
                     'gr_last_added_density_myofiber',
                     'gr_last_added_density_ground_matrix',
                     'gr_last_added_density_collagen',
                     'gr_growth_volume_ratio',
                     'gr_geometric_volume_ratio'):
            if name not in self.data:
                continue
            canonical = dict(zip(self.material_point_keys, self.data[name]))
            self.data[name] = np.asarray(
                [canonical[key] for key in target_keys])

    def _integrate_constituent_masses(self, mesh_object):
        domain = mesh_object.model['mesh']
        result = {}
        for constituent in ('myofiber', 'ground_matrix', 'collagen'):
            field = mesh_object.model['functions'][
                'constituent_density_' + constituent]
            result[constituent] = assemble(
                field*dx(domain=domain),
                form_compiler_parameters={'representation': 'uflacs'})
        result['total'] = sum(result.values())
        return result

    def finalize_constituent_mass_event(self, mesh_object, event_diagnostics):
        """Integrate physical masses on the moved reference mesh."""
        new_mass = self._integrate_constituent_masses(mesh_object)
        old_mass = self.global_constituent_mass
        added = event_diagnostics['added_constituent_mass']
        residuals = dict(
            (name, new_mass[name] - old_mass[name] - added[name])
            for name in ('myofiber', 'ground_matrix', 'collagen'))
        residuals['total'] = new_mass['total'] - old_mass['total'] - \
            sum(added.values())
        self.global_constituent_mass = new_mass
        self.data['gr_global_mass_myofiber'] = new_mass['myofiber']
        self.data['gr_global_mass_ground_matrix'] = new_mass['ground_matrix']
        self.data['gr_global_mass_collagen'] = new_mass['collagen']
        self.data['gr_global_mass_total'] = new_mass['total']
        self.data['gr_mass_balance_residual'] = residuals['total']
        self.data['gr_mass_balance_residual_myofiber'] = \
            residuals['myofiber']
        self.data['gr_mass_balance_residual_ground_matrix'] = \
            residuals['ground_matrix']
        self.data['gr_mass_balance_residual_collagen'] = \
            residuals['collagen']
        self.data['gr_pre_growth_myocardial_volume'] = \
            event_diagnostics['pre_growth_volume']
        self.data['gr_post_growth_myocardial_volume'] = \
            event_diagnostics['post_growth_volume']
        self.data['gr_prescribed_positive_growth_volume'] = \
            event_diagnostics['prescribed_positive_growth_volume']
        self.data['gr_growth_event_number'] = self.growth_event_number
        scale = max(1.0, abs(new_mass['total']))
        tolerance = float(self.mass_bookkeeping_config.get(
            'mass_balance_tolerance', 1.0e-6))
        if max(abs(value) for value in residuals.values()) > tolerance*scale:
            raise RuntimeError(
                'Global constituent mass balance failed after ALE growth')
        if self.comm.Get_rank() == 0:
            print '[GrowthConstituentMass] event', self.growth_event_number
            print '  pre/post myocardial volume:', \
                event_diagnostics['pre_growth_volume'], \
                event_diagnostics['post_growth_volume']
            print '  prescribed positive growth volume:', \
                event_diagnostics['prescribed_positive_growth_volume']
            print '  global masses:', new_mass
            print '  mass-balance residuals:', residuals
    
    def store_setpoint(self,store_active):
        """ Store setpoint data before growth activation """
        if store_active:
            for comp in self.components:
                s = comp.return_signal()
                comp.data['setpoint_tracker'].append(s)
                #comp.store_setpoint()

    def implement_growth(self,end_diastolic,time_step):
        
        if self.data['gr_active']:
            # First update mesh class
            self.mesh = self.parent_circulation.mesh

            # Then,update the stimulus and theta for each growth type
            for comp in self.components:
   
                set_name = 'gr_setpoint' + '_' +  comp.data['type']
                comp.data['setpoint'] = self.data[set_name] 

                # update stimulus signal 
                if self.data['gr_stimulus_active']:
                    s = comp.return_signal()
                    comp.data['stimulus_tracker'].append(s)
                    #comp.store_stimulus()

                # now save values over dolfin functions to visualize
                for value in ['stimulus','setpoint','deviation']:
                    name = value + '_' + comp.data['type']
                    self.parent_circulation.mesh.model['functions'][name].vector()[:] = \
                        comp.data[value]
                # save theta values
                for theta in ['local_theta','global_theta']:
                    theta_name = theta+'_vis_' + comp.data['type']
                    self.parent_circulation.mesh.model['functions'][theta_name].vector()[:] = \
                            comp.data[theta]
                
                if end_diastolic:
                    if self.growth_frequency_n_counter == self.growth_frequency_n:

                        if self.comm.Get_rank() == 0:
                            print('Growth is happening at ED!')
                        
                        comp.data['stimulus'] = \
                            np.mean(comp.data['stimulus_tracker'],axis=0)
                        if self.comm.Get_rank() == 0: 
                            print 'len stimulus_tracker before reseting'
                            print len(comp.data['stimulus_tracker'])
                        comp.data['stimulus_tracker'] = []

                        comp.data['deviation'] = \
                            comp.data['stimulus'] - comp.data['setpoint']
                        # update thetas per cycle 
                        comp.data['local_theta'] = \
                            comp.return_local_theta(comp.data['global_theta'],
                                                    comp.data['stimulus'],
                                                    comp.data['setpoint'])
                        if comp.data['type'] in self.prescribed_growth_increment:
                            comp.data['local_theta'][:] = \
                                self.prescribed_growth_increment[
                                    comp.data['type']]
                        
                        new_global_theta = np.ones(len(comp.data['local_theta']))
                        for i,t in enumerate(comp.data['local_theta']):

                            if comp.data['local_theta'][i]> comp.data['local_theta_max']:
                                comp.data['local_theta'][i] =comp.data['local_theta_max']
                            if comp.data['local_theta'][i] < comp.data['local_theta_min']:
                                comp.data['local_theta'][i] = comp.data['local_theta_min']

                            new_global_theta[i] = \
                                comp.data['global_theta'][i]*t
                                
                            if new_global_theta[i] > comp.data['theta_max']:
                                new_global_theta[i] = comp.data['theta_max']
                            if new_global_theta[i] < comp.data['theta_min']:
                                new_global_theta[i] = comp.data['theta_min']
                        comp.data['global_theta'] = new_global_theta
  
                        # Update theta functions to update Fg
                        name = 'temp_theta_' + comp.data['type']
                        self.mechan.model['functions'][name].vector()[:] = \
                            comp.data['local_theta']


                for param in ['local_theta','global_theta','setpoint','stimulus','deviation']:
                    name = 'gr_' + param + '_'+  comp.data['type']
                    self.data[name] = comp.data[param]

    def grow_reference_config(self):

        # growth the mesh with Fg
        if self.comm.Get_rank() == 0:
            print 'Solveing Fg = 0'
        self.solver.solve_growth()
        #self.solver.solvenonlinear()
        Fg = self.mechan.model['functions']['Fg']
                  
        Fe = self.mechan.model['functions']['Fe']
                        #Fg = self.mesh.model['functions']['Fg']
        temp_Fg = project(Fg,self.mechan.model['function_spaces']['tensor_space'],
                            form_compiler_parameters={"representation":"uflacs"}).vector().get_local()[:]
        temp_Fe = project(Fe,self.mechan.model['function_spaces']['tensor_space'],
                            form_compiler_parameters={"representation":"uflacs"}).vector().get_local()[:]
        F = self.mechan.model['functions']['Fmat']
        temp_F = project(F,self.mechan.model['function_spaces']['tensor_space'],
                        form_compiler_parameters={"representation":"uflacs"}).vector().get_local()[:]
        vol = assemble(1.0*dx(domain = self.mechan.model['mesh']), form_compiler_parameters={"representation":"uflacs"})
        pre_growth_volume = float(vol)
        movement = project(
            self.mechan.model['functions']['w'].sub(0),
            VectorFunctionSpace(self.mechan.model['mesh'], 'CG', 1),
            form_compiler_parameters={'representation': 'uflacs'})
        movement_gradient = Identity(
            self.mechan.model['mesh'].geometry().dim()) + grad(movement)
        geometric_ratio_function = project(
            det(movement_gradient),
            self.mechan.model['function_spaces']['quadrature_space'],
            form_compiler_parameters={'representation': 'uflacs'})
        geometric_volume_ratios = self._values_in_canonical_order(
            self.mechan,
            geometric_ratio_function.vector().get_local().tolist())

        event_diagnostics = None
        if self.constituent_mass_bookkeeper is not None:
            theta_f = self.mechan.model['functions']['theta_fiber']
            theta_s = self.mechan.model['functions']['theta_sheet']
            theta_n = self.mechan.model['functions']['theta_sheet_normal']
            Jg = theta_f*theta_s*theta_n
            positive_growth = conditional(Jg > 1.0, Jg - 1.0, 0.0)
            density = Constant(self.mass_bookkeeping_config['density'])
            passive = self.mechan.model['functions']['dolfin_functions'][
                'passive_params']
            if self.mass_bookkeeping_config['mode'] == 'proportional':
                allocations = {
                    'myofiber': passive['phi_m'][-1],
                    'ground_matrix': passive['phi_g'][-1],
                    'collagen': passive['phi_c'][-1],
                }
            else:
                allocations = dict(
                    (name, Constant(weight)) for name, weight in
                    self.mass_bookkeeping_config['weights'].items())
            added_constituent_mass = dict(
                (name, float(assemble(
                    density*positive_growth*allocation*
                    dx(domain=self.mechan.model['mesh']),
                    form_compiler_parameters={'representation': 'uflacs'})))
                for name, allocation in allocations.items())
            event_diagnostics = {
                'pre_growth_volume': pre_growth_volume,
                'prescribed_positive_growth_volume': float(assemble(
                    positive_growth*dx(domain=self.mechan.model['mesh']),
                    form_compiler_parameters={'representation': 'uflacs'})),
                'added_constituent_mass': added_constituent_mass,
                'geometric_volume_ratios': geometric_volume_ratios,
            }
        #temp_vol = self.mesh.model['uflforms'].LVcavityvol()
        #if self.comm.Get_rank() == 0:
        #    print "LV volume: %f" %temp_vol
        if self.comm.Get_rank() == 0:
            print 'Fg after solving for growth, but before ALE'
            print temp_Fg
            print 'Fe after solving for growth, but before ALE'
            print temp_Fe
            print 'F after solving for growth, but before ALE'
            print temp_F
            print 'dx before'
            print vol
        
        # move the mesh and build up new reference config
        (u,p,c11)   = split(self.mechan.model['functions']['w'])
        #print 'u values'
        #print self.mechan.model['functions']['w'].vector().get_local()[:]
        mesh = self.mechan.model['mesh']
        if self.comm.Get_rank() == 0:
            print 'Moving reference mesh'
        ALE.move(self.mechan.model['mesh'], movement)
        Fg = self.mechan.model['functions']['Fg']
                  
        Fe = self.mechan.model['functions']['Fe']
                        #Fg = self.mesh.model['functions']['Fg']
        temp_Fg = project(Fg,self.mechan.model['function_spaces']['tensor_space'],
                            form_compiler_parameters={"representation":"uflacs"}).vector().get_local()[:]
        temp_Fe = project(Fe,self.mechan.model['function_spaces']['tensor_space'],
                            form_compiler_parameters={"representation":"uflacs"}).vector().get_local()[:]
        F = self.mechan.model['functions']['Fmat']
        temp_F = project(F,self.mechan.model['function_spaces']['tensor_space'],
                        form_compiler_parameters={"representation":"uflacs"}).vector().get_local()[:]
        vol = assemble(1.0*dx(domain = self.mechan.model['mesh']), form_compiler_parameters={"representation":"uflacs"})
        if event_diagnostics is not None:
            event_diagnostics['post_growth_volume'] = float(vol)
        if self.comm.Get_rank() == 0:
            print 'Fg after solving for growth and after ALE'
            print temp_Fg
            print 'Fe after solving for growth and after ALE'
            print temp_Fe
            print 'F after solving for growth and after ALE'
            print temp_F
            print 'dx after'
            print vol
        return event_diagnostics
      
    def reinitialize_mesh_object_for_growth(self,
                                            predefined_functions):
        predefined_mesh = self.mechan.model['mesh']
        # re-create object for the mechanics 
        self.mechan = GrowthMechanicsClass(self.parent_circulation,
                                            predefined_mesh=predefined_mesh,
                                            predefined_functions=predefined_functions)
        # create object for the solver of the growth
        self.solver = NSolver(self.parent_circulation,self.mechan,self.comm)
class growth_component():
    "Class for a growth component"

    def __init__(self,comp_struct,parent_growth):

        self.parent = parent_growth
        self.data = dict()
        for item in comp_struct:
            self.data[item] = comp_struct[item][0]
        for required in ('type', 'signal', 'tau', 'theta_min', 'theta_max',
                         'local_theta_min', 'local_theta_max'):
            if required not in self.data:
                raise ValueError(
                    'Growth component requires model.growth.components.%s' %
                    required)
        theta_name = 'theta' + '_' + self.data['type']
        size = len(self.parent.mechan.model['functions'][theta_name].vector().get_local()[:])

        for item in ['stimulus','stimulus_tracker',
                    'local_theta', 'global_theta',
                    'setpoint', 'setpoint_tracker', 
                    'deviation']:
            if item  in ['local_theta','global_theta']:
                self.data[item] = np.ones(size)
            elif item  in ['stimulus','setpoint','deviation']:
                self.data[item] = np.zeros(size)
            else:
                self.data[item] = []
        
    def return_local_theta(self,global_theta,stimulus,setpoint):
        """ Update local thetha values wrt previous growth step"""

        local_theta = np.ones(len(global_theta))
        if self.parent.comm.Get_rank() == 0:
            print 'local_theta'
            print local_theta
        for i,s in enumerate(stimulus):
            theta_g = global_theta[i]
            set_p = setpoint[i]
            dtheta = self.return_delta_theta(theta_g,s,set_p)
            local_theta[i] += dtheta
        if self.parent.comm.Get_rank() == 0:
            print 'local_theta new'
            print local_theta
        return local_theta
    
    def return_delta_theta(self,global_theta,s,setpoint):
        """ Return delta local theta"""
        # theta merges to thta_max if s > set and vice versa.
        tau = float(self.data['tau'])
        range_theta = self.data['theta_max'] - self.data['theta_min']
        set_component = (s - setpoint)/np.amax([np.abs(setpoint),1])
        if s-setpoint>=0:
            dtheta = (set_component/tau)*(self.data['theta_max'] - global_theta)/range_theta
        else:

            dtheta = (set_component/tau)*(global_theta - self.data['theta_min'])/range_theta
        return dtheta
    
    def return_signal(self):
        """ Calculate stimulus instantaneous signal  """
        
        if self.parent.comm.Get_rank() == 0:
            print 'r\Returning stimulus signal!'
                        


        """hsl = self.parent.mesh.model['functions']['hsl']
        f0 = self.parent.mesh.model['functions']['f0']
        scalar_fs = self.parent.mesh.model['function_spaces']['growth_scalar_FS']
        total_passive,myofiber_passive = \
                self.parent.mesh.model['uflforms'].stress(hsl)"""

        if self.data['signal'] == 'myofiber_passive_stress':
            s = project(inner(self.parent.mesh.model['functions']['f0'],
                            self.parent.mesh.model['functions']['Sff']*self.parent.mesh.model['functions']['f0']),
                            self.parent.mesh.model['function_spaces']['quadrature_space']).vector().array()[:]
            
        if self.data['signal'] == 'total_stress':
            #active_stress = self.parent.mesh.model['functions']['Pactive']
            #total_stress = total_passive + active_stress
            inner_p = inner(self.parent.mesh.model['functions']['f0'],
                        self.parent.mesh.model['functions']['total_stress']*self.parent.mesh.model['functions']['f0'])

            s_projected = project(inner_p,self.parent.mesh.model['function_spaces']['growth_scalar_FS'],
                            form_compiler_parameters={"representation":"uflacs"})
            if self.parent.comm.Get_rank() == 0:
                print 'projected total stress'
                print s_projected.vector().get_local()[:]
                print 'size'
                print len(s_projected.vector().get_local()[:])
            s = interpolate(s_projected,self.parent.mesh.model['function_spaces']['quadrature_space']).vector().get_local()[:]
            if self.parent.comm.Get_rank() == 0:    
                print 'interpolated total stress'
                print s
                print 'size'
                print len(s)
        return s
