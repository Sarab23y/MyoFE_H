"""Independent DOLFIN reader for one full mixed-state diagnostic checkpoint.

Run in the FEniCS image at both 1 and 32 ranks. Coordinate correspondence, not
rank-local array equality, matches the writer's quadrature values.
"""
from __future__ import print_function
import argparse
import glob
import json
import os
import sys
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    return run(args)


def run(args):
    repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,os.path.join(repo,'python_codes'))
    from dolfin import (Mesh, HDF5File, MPI, VectorElement, FiniteElement,
                        MixedElement, FunctionSpace, Function, split, Identity, grad,
                        parameters)
    from mpi4py import MPI as MP
    from diagnostic_io import open_hdf5_file, root_call, write_report
    from LV_simulation.dependencies.fenics_2017_compat import project_quadrature
    from LV_simulation.dependencies.fe_cycle_core import deformation_metrics
    comm=MP.COMM_WORLD
    if comm.Get_size()>1:
        parameters['mesh_partitioner']='SCOTCH'
    parameters['form_compiler']['quadrature_degree']=2
    parameters['form_compiler']['representation']='quadrature'
    mesh=Mesh(MPI.comm_world)
    handle=open_hdf5_file(HDF5File,mesh.mpi_comm(),args.state,'r')
    try:
        handle.read(mesh,'/mesh',False)
        cell=mesh.ufl_cell()
        displacement=VectorElement('CG',cell,2,quad_scheme='default')
        pressure=FiniteElement('CG',cell,1,quad_scheme='default')
        real=FiniteElement('Real',cell,0)
        W=FunctionSpace(mesh,MixedElement([displacement,pressure,real,MixedElement([real]*5)]))
        metadata=os.path.join(os.path.dirname(os.path.dirname(args.state)),'mapping.json')
        if os.path.isfile(metadata):
            with open(metadata) as source:
                expected=json.load(source).get('function_space')
            if expected and expected!=str(W.ufl_element()):
                raise ValueError('reconstructed mixed element differs from writer metadata')
        w=Function(W);handle.read(w,'/w')
        quad=FiniteElement('Quadrature',cell,2,quad_scheme='default')
        Q=FunctionSpace(mesh,quad)
        VQ=FunctionSpace(mesh,VectorElement('Quadrature',cell,2,quad_scheme='default'))
        f0=Function(VQ);handle.read(f0,'/f0')
        hsl0=Function(Q);handle.read(hsl0,'/hsl0')
    finally:
        handle.close()
    F=Identity(3)+grad(split(w)[0])
    def value(expression,name):
        return project_quadrature(expression,Q,'reopen_'+name,comm).vector().get_local()
    fields=np.column_stack([value(F[i,j],'F%d%d'%(i,j)) for i in range(3) for j in range(3)]).reshape((-1,3,3))
    fiber=np.column_stack([value(f0[i],'f%d'%i) for i in range(3)])
    metrics=deformation_metrics(fields,fiber,hsl0.vector().get_local())
    coordinates=Q.tabulate_dof_coordinates().reshape((-1,3))[:len(fields)]
    gathered=comm.gather(dict(coordinates=coordinates,F=fields,J=metrics['J'],stretch=metrics['stretch'],hsl=metrics['hsl']),root=0)
    def evaluate():
        from scipy.spatial import cKDTree
        directory=os.path.dirname(args.state)
        observer=os.path.dirname(directory)
        step=int(os.path.basename(args.state).split('_')[1].split('.')[0])
        references=[]
        for path in sorted(glob.glob(os.path.join(directory,'*_%08d_rank*.npz'%step))):
            rank=int(os.path.basename(path).split('rank')[1].split('.')[0])
            with np.load(os.path.join(observer,'mapping_rank%04d.npz'%rank)) as mapping, np.load(path) as data:
                if 'F' not in data:
                    continue
                if not np.array_equal(mapping['global_ids'],data['global_ids']):
                    raise ValueError('writer frame/mapping IDs differ')
                references.append(dict(coordinates=mapping['coordinates'].copy(),
                    F=data['F'].copy(),J=data['J'].copy(),stretch=data['stretch'].copy(),hsl=data['hsl'].copy()))
        if not references:
            raise ValueError('no reference F frames for requested state')
        # A checkpoint and an event NPZ may refer to the same frame. Reject
        # inconsistent duplicates rather than double-counting material points.
        by_coordinate={}
        for part in references:
            for index,coordinate in enumerate(part['coordinates']):
                key=tuple(coordinate)
                item=dict((name,part[name][index]) for name in ('F','J','stretch','hsl'))
                if key in by_coordinate:
                    for name in item:
                        if not np.array_equal(item[name],by_coordinate[key][name]):
                            raise ValueError('inconsistent duplicate writer frame')
                by_coordinate[key]=item
        expected=np.asarray(sorted(by_coordinate))
        actual=np.concatenate([p['coordinates'] for p in gathered])
        distance,indices=cKDTree(expected).query(actual)
        if len(actual)!=len(expected) or len(set(indices.tolist()))!=len(expected) or np.max(distance)>1.e-9:
            raise ValueError('reopened quadrature coordinates do not match uniquely')
        errors={}
        for name in ('F','J','stretch','hsl'):
            values=np.concatenate([p[name] for p in gathered])
            reference=np.asarray([by_coordinate[tuple(expected[i])][name] for i in indices])
            errors[name]=float(np.max(np.abs(values-reference)))
            if not np.allclose(values,reference,atol=1.e-8,rtol=1.e-9):
                raise ValueError('independent checkpoint reconstruction mismatch: '+name)
        result=dict(passed=True,ranks=comm.Get_size(),points=len(actual),errors=errors,
                    maximum_coordinate_error=float(np.max(distance)),
                    state_path=os.path.realpath(args.state),state_step=step,
                    mixed_element=str(W.ufl_element()),
                    full_coupled_restart=False,full_pipeline_validated=False)
        import hashlib
        digest=hashlib.sha256()
        with open(args.state,'rb') as handle:
            for block in iter(lambda:handle.read(1048576),b''):digest.update(block)
        result['state_sha256']=digest.hexdigest()
        write_report(args.output,result)
        return result
    result=root_call(comm,evaluate)
    if comm.Get_rank()==0:print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
