# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:15:59 2022

@author: Hossein
"""

import sys
import os
import json
import time as TIME
import warnings

from LV_simulation.dependencies.recode_dictionary import recode
from LV_simulation.LV_simulation import LV_simulation as lvs
from LV_simulation.output_handler import output_handler as oh

from dolfin import *
from mpi4py import MPI
import cProfile
import pstats

warnings.filterwarnings("ignore", category=FutureWarning)


def MyoFE():

    # start counting simulation time
    start = TIME.time()

    # Get the comminicator 
    comm = MPI.COMM_WORLD

    # Determine the number of cores have been called (size)
    # and core id number (rank) 
    size = comm.Get_size()
    rank = comm.Get_rank()

    """#SARA: I ADD IT ON 6/25/2025
    print "[MPI] Core %d of %d is active." % (rank+1, size)"""

    if rank == 0:
        print ('%.0f numbers of core is called!' %size)
                  
    # get number of arguments
    no_of_arguments = len(sys.argv)

    debug_mpi = False

    if no_of_arguments == 1:
        print "No argument is called"

    # handle if only simulation type has been called 
    # by running a demo
    elif no_of_arguments == 2:
        if sys.argv[1] == 'LV_sim':
            instruct_file = ''

    # handle if  a simulation has been called with an 
    # instruction file

    elif no_of_arguments == 3:
        if sys.argv[1] == 'LV_sim':
            instruct_file = sys.argv[2]
            try:
                debug_mpi = execute_MyoFE(instruct_file, comm)
            except Exception as e:
                print "Fatal error:", str(e)
                MPI.COMM_WORLD.Abort(1)

            """#SARA ADDED ON 6/25/2025
            try:
                comm.Barrier()
                execute_MyoFE(instruct_file, comm)
                comm.Barrier()
            except Exception as e:
                print "[Core %d] Error: %s" % (rank, str(e))
                raise  # re-throw the error so MPI terminates properly"""

    stop = TIME.time()
    print 'Batch run time'
    dt = stop-start
    print dt

    if debug_mpi:
        print '[MPI DEBUG] rank=%d comm=%s finalized=%s' % \
            (rank, str(type(comm)), str(MPI.Is_finalized()))
    
def execute_MyoFE(instruction_file,comm):
    
    # first load instruction file
    with open(instruction_file, 'r') as f:
           instruction_data = json.load(f)

    # and then run the simulation
    instruction_data = recode(instruction_data)
    debug_mpi = False
    if 'debug_mpi' in instruction_data:
        try:
            debug_mpi = bool(instruction_data['debug_mpi'][0])
        except:
            debug_mpi = bool(instruction_data['debug_mpi'])

    if debug_mpi:
        print '[MPI DEBUG] rank=%d comm=%s finalized=%s' % \
            (comm.Get_rank(), str(type(comm)), str(MPI.Is_finalized()))

    prot_struct = instruction_data['protocol']
    output_struct = []
    if 'output_handler' in instruction_data:
        output_struct = instruction_data['output_handler']
  
    LV_sim_object = lvs(comm,instruction_data = instruction_data)
    LV_sim_object.run_simulation(protocol_struct = prot_struct,
                                    output_struct = output_struct)
    return debug_mpi






'''def profile_MyoFE():
    cProfile.runctx('MyoFE()', globals(), locals(), filename='profiling_results_MyoFE.txt')
profile_MyoFE()
# Get the full path of the profiling results file
file_path = os.path.join(os.getcwd(), 'profiling_results_MyoFE.txt')
# Load the profiling results
stats = pstats.Stats(file_path)

# Strip directory names from filenames
stats.strip_dirs()
# Sort the statistics by cumulative time
stats.sort_stats('cumulative')
# Get the top 10 functions based on cumulative time
stats.print_stats(100)
'''



if __name__ == '__main__':
    MyoFE()
