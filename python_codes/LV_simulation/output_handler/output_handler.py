# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:54:34 2022

@author: Hossein
"""

import json
import os

import numpy as np
import pandas as pd


class output_handler():
    """ Class for handling simulation output """

    def __init__(self, output_struct):


        self.total_file_disp = []
        self.sim_data_file_str = []
        self.images_handler_list = []
        self.output_data_str = []
        # Central CSV name registry for JSON-controlled tabular output selection.
        # This does not alter mesh outputs, which keep their native formats.
        self.available_csv_outputs = {
            'data.csv': 'Main simulation CSV'
        }
        self.selected_csv_outputs = set(['data.csv'])

        # Check for output_handler file
        if (output_struct == []):
            print('No output handler file specified. Cannot write output')
            return

        if 'output_mesh_file' in output_struct:
            output_mesh_str = output_struct['output_mesh_file'][0]
            self.check_output_directory_folder(path = output_mesh_str)
            self.total_file_disp = File(output_mesh_str)

        if 'output_data_path' in output_struct:
            self.output_data_str = output_struct['output_data_path'][0]
            self.check_output_directory_folder(path = self.output_data_str)
            if 'output_excel_path' in output_struct:
                print('Ignoring output_excel_path; CSV-only output mode is enforced.')

        self.configure_output_selection(output_struct)

    



        # Load the output handler structure as a dict
        #with open(output_handler_file_string, 'r') as f:
        #    self.oh_data = json.load(f)

        

        
    def check_output_directory_folder(self, path=""):
        """ Check output folder"""
        output_dir = os.path.dirname(path)
        print('output_dir %s' % output_dir)
        if not os.path.isdir(output_dir):
            print('Making output dir')
            os.makedirs(output_dir)


    def wrap_up_simulation(self,
                            sim_data):

        if not isinstance(sim_data, pd.DataFrame):
                print('No simulation data available')
                return
        # First save data if it is called
        if self.output_data_str and self.should_save_output('data.csv'):
            sim_data.to_csv(self.output_data_str)
        if self.output_excel_str and self.should_save_output('data.xlsx'):
            sim_data.to_excel(self.output_excel_str, index=False)

        # Then generate figures if any
        
        return

    def configure_output_selection(self, output_struct):
        requested = None
        if output_struct and ('save_outputs' in output_struct):
            requested = output_struct['save_outputs']

        if requested is None:
            self.selected_csv_outputs = set(['data.csv'])
            return

        if isinstance(requested, str):
            if requested == 'all':
                self.selected_csv_outputs = set(self.available_csv_outputs.keys())
                return
            if requested == 'data.xlsx':
                print('Mapping legacy save_outputs entry "data.xlsx" to "data.csv" (CSV-only mode).')
                requested = 'data.csv'
            requested = [requested]

        if isinstance(requested, list) and ('all' in requested):
            self.selected_csv_outputs = set(self.available_csv_outputs.keys())
            return

        if isinstance(requested, list):
            remapped = []
            for entry in requested:
                if entry == 'data.xlsx':
                    print('Mapping legacy save_outputs entry "data.xlsx" to "data.csv" (CSV-only mode).')
                    remapped.append('data.csv')
                else:
                    remapped.append(entry)
            requested = remapped

        if not isinstance(requested, list):
            raise ValueError('output_handler.save_outputs must be "all" or a list of output names')

        invalid = [x for x in requested if x not in self.available_csv_outputs]
        if invalid:
            valid = sorted(self.available_csv_outputs.keys())
            raise ValueError('Invalid save_outputs entries: %s. Valid options: %s' % (str(invalid), str(valid)))
        self.selected_csv_outputs = set(requested)

    def should_save_output(self, output_name, config=None):
        selected = self.selected_csv_outputs if config is None else set(config)
        return output_name in selected
