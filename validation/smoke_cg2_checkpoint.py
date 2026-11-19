"""Actual DOLFIN manufactured CG2 checkpoint test on the actual input mesh.

This does not evolve the LV or create a coupled restart. Execute at 1/32 ranks.
"""
from __future__ import print_function
import argparse
import json
import os
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-json',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    import dolfin as df
    from mpi4py import MPI
    from diagnostic_io import root_call, open_hdf5_file, write_report
    from mpi_compat import world_comm
    from validate_myosim_mixed_quadrature_ordering import _production_mesh
    import validate_fe_cycle_checkpoint as reader
    repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,os.path.join(repo,'python_codes'))
    from LV_simulation.dependencies.fe_cycle_core import deformation_metrics
    from LV_simulation.dependencies.fenics_2017_compat import project_quadrature
    comm=MPI.COMM_WORLD
    if comm.Get_size()>1:df.parameters['mesh_partitioner']='SCOTCH'
    df.parameters['form_compiler']['quadrature_degree']=2
    df.parameters['form_compiler']['representation']='quadrature'
    def create():
        os.makedirs(args.output);os.mkdir(os.path.join(args.output,'frames'))
    root_call(comm,create)
    mesh,path=_production_mesh(args.input_json,repo,(df.HDF5File,df.Mesh,lambda:world_comm(df)))
    cell=mesh.ufl_cell()
    ve=df.VectorElement('CG',cell,2,quad_scheme='default');pe=df.FiniteElement('CG',cell,1,quad_scheme='default');re=df.FiniteElement('Real',cell,0)
    W=df.FunctionSpace(mesh,df.MixedElement([ve,pe,re,df.MixedElement([re]*5)]))
    V=df.FunctionSpace(mesh,ve)
    u=df.interpolate(df.Expression(('0.02*x[0]*x[1]','-0.01*x[1]*x[2]','0.015*x[0]*x[2]'),degree=2),V)
    w=df.Function(W);df.assign(w.sub(0),u)
    V1=df.VectorFunctionSpace(mesh,'CG',1)
    linearized=df.interpolate(df.interpolate(u,V1),V)
    local=float(np.max(np.abs(u.vector().get_local()-linearized.vector().get_local())))
    vertex_only_error=max(comm.allgather(local))
    if vertex_only_error<=1.e-12:raise RuntimeError('fixture fails to exercise CG2 edge information')
    quad=df.FiniteElement('Quadrature',cell,2,quad_scheme='default')
    Q=df.FunctionSpace(mesh,quad)
    VQ=df.FunctionSpace(mesh,df.VectorElement('Quadrature',cell,2,quad_scheme='default'))
    f0=df.Function(VQ)
    original=open_hdf5_file(df.HDF5File,mesh.mpi_comm(),path,'r')
    try:original.read(f0,'ellipsoidal/eF')
    finally:original.close()
    hsl0=df.interpolate(df.Constant(1000.),Q)  # manufactured kinematic scale, not historical
    F=df.Identity(3)+df.grad(df.split(w)[0])
    def values(expr,name):return project_quadrature(expr,Q,'cg2_smoke_'+name,comm).vector().get_local()
    tensors=np.column_stack([values(F[i,j],'F%d%d'%(i,j)) for i in range(3) for j in range(3)]).reshape((-1,3,3))
    fiber=np.column_stack([values(f0[i],'f%d'%i) for i in range(3)])
    metrics=deformation_metrics(tensors,fiber,hsl0.vector().get_local())
    root_call(comm,lambda:write_report(os.path.join(args.output,'mapping.json'),dict(
        function_space=str(W.ufl_element()),manufactured_field=True,
        original_mesh_path=path,mixed_global_dimension=W.dim())))
    ids=np.asarray([Q.dofmap().local_to_global_index(i) for i in range(len(tensors))])
    np.savez(os.path.join(args.output,'mapping_rank%04d.npz'%comm.Get_rank()),global_ids=ids,
        coordinates=Q.tabulate_dof_coordinates().reshape((-1,3))[:len(ids)])
    np.savez(os.path.join(args.output,'frames/checkpoint_00000000_rank%04d.npz'%comm.Get_rank()),
        global_ids=ids,F=tensors,J=metrics['J'],stretch=metrics['stretch'],hsl=metrics['hsl'])
    target=os.path.join(args.output,'frames/state_00000000.h5')
    writer=open_hdf5_file(df.HDF5File,mesh.mpi_comm(),target,'w')
    try:
        writer.write(mesh,'/mesh');writer.write(w,'/w');writer.write(f0,'/f0');writer.write(hsl0,'/hsl0')
    finally:writer.close()
    reopened=df.Function(W)
    handle=open_hdf5_file(df.HDF5File,mesh.mpi_comm(),target,'r')
    try:handle.read(reopened,'/w')
    finally:handle.close()
    exact=all(comm.allgather(bool(np.array_equal(w.vector().get_local(),reopened.vector().get_local()))))
    if not exact:raise RuntimeError('same-space mixed coefficients did not roundtrip exactly')
    comm.Barrier()
    reader.run(argparse.Namespace(state=target,output=os.path.join(args.output,'reconstruction.json')))
    root_call(comm,lambda:write_report(os.path.join(args.output,'checkpoint_smoke.json'),dict(
        passed=True,dolfin_version=df.__version__,mpi_size=comm.Get_size(),manufactured_field=True,actual_input_mesh=True,
        exact_mixed_coefficient_roundtrip=exact,cg2_vs_vertex_linear_error=vertex_only_error,
        displacement_element=str(ve),mixed_element=str(W.ufl_element()),
        full_coupled_restart=False,full_pipeline_validated=False)))


if __name__=='__main__':main()
