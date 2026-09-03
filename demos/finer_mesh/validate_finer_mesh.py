from __future__ import print_function
from dolfin import *

mesh_path = '../demos/finer_mesh/ellipsoidal.hdf5'
dataset_name = 'ellipsoidal'

mesh = Mesh()
h5 = HDF5File(mpi_comm_world(), mesh_path, 'r')
h5.read(mesh, dataset_name, False)
h5.close()

print('mesh path:', mesh_path)
print('dataset:', dataset_name)
print('vertices:', mesh.num_vertices())
print('cells:', mesh.num_cells())
print('cell type:', mesh.ufl_cell())
