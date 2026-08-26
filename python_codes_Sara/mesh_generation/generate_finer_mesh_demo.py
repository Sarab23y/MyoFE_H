from __future__ import print_function
import os
from dolfin import *
from Ellipsoidal_LV import create_ellipsoidal_LV, EllipsoidalLVMEsh, check_output_directory_folder

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    outdir = os.path.join(repo, 'demos', 'finer_mesh') + '/'
    check_output_directory_folder(path=outdir)

    input_geo_file = os.path.join(here, 'ellipsoidal_thin_apex.geo')
    vtk_file_name = 'Ellipsoidal_finer'
    temp_vtk_dir = os.path.join(here, 'input_files_finer')
    check_output_directory_folder(path=temp_vtk_dir)

    create_ellipsoidal_LV(
        geofile=input_geo_file,
        output_vtk=temp_vtk_dir,
        casename=vtk_file_name,
        meshsize=0.06,
        gmshcmd='gmsh',
        iswritemesh=True,
        verbose=False)

    vtk_file_str = os.path.join(temp_vtk_dir, vtk_file_name + '.vtk')
    EllipsoidalLVMEsh(
        vtk_file_str=vtk_file_str,
        output_file_str=outdir,
        quad_deg=2,
        endo_angle=60,
        epi_angle=-60,
        endo_hsl=900,
        epi_hsl=1000)

    mesh_path = os.path.join(outdir, 'ellipsoidal.hdf5')
    with open(mesh_path, 'rb') as f:
        sig = f.read(8)
    if sig != b'\x89HDF\r\n\x1a\n':
        raise RuntimeError('Generated file is not HDF5: %r' % (sig,))

    m = Mesh()
    h5 = HDF5File(mpi_comm_world(), mesh_path, 'r')
    h5.read(m, 'ellipsoidal', False)
    h5.close()
    print('Generated mesh:', mesh_path)
    print('Vertices:', m.num_vertices())
    print('Cells:', m.num_cells())

if __name__ == '__main__':
    main()
