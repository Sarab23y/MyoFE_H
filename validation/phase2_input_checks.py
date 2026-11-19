"""Read-only candidate mesh/basis checks; no MPI or producer authentication."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree
from FIAT.reference_element import UFCTetrahedron
from FIAT.quadrature_schemes import create_quadrature


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mesh',type=Path,required=True);ap.add_argument('--coordinates',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    fields={};cell_ids=None
    with h5py.File(str(a.mesh),'r') as h:
        X=h['ellipsoidal/coordinates'][:];T=h['ellipsoidal/topology'][:]
        for name in ('eF','eS','eN'):
            g=h['ellipsoidal/'+name];signature=g.attrs['signature']
            if isinstance(signature,bytes):signature=signature.decode()
            if signature!="VectorElement(FiniteElement('Quadrature', tetrahedron, 2, quad_scheme='default'), dim=3)":raise ValueError('Unsupported basis signature')
            offset=g['x_cell_dofs'][:];ids=g['cell_dofs'][:];values=g['vector_0'][:];cells=g['cells'][:]
            if cell_ids is not None and not np.array_equal(cell_ids,cells):raise ValueError('Reference basis cell orders differ')
            cell_ids=cells
            if not np.all(np.diff(offset)==12):raise ValueError('Unexpected local vector Quadrature dimension')
            fields[name]=np.asarray([values[ids[i:j]].reshape(3,4).T for i,j in zip(offset[:-1],offset[1:])]).reshape(-1,3)
    q=create_quadrature(UFCTetrahedron(),2,'default').get_points();B=(X[T[:,1:]]-X[T[:,0,None]]).transpose(0,2,1)
    coordinates=(np.einsum('cij,qj->cqi',B,q)+X[T[:,0,None]]).reshape(-1,3);saved=np.load(str(a.coordinates));distance,index=cKDTree(coordinates).query(saved)
    geometry=dict(evidence_class='candidate coordinate set only; not MPI IDs',candidate_points=len(coordinates),saved_points=len(saved),maximum_nearest_distance=float(distance.max()),unique_candidate_matches=len(set(index.tolist())),coordinate_set_matches=bool(distance.max()<1e-10 and len(set(index.tolist()))==len(coordinates)))
    basis=dict(evidence_class='candidate HDF5 cell DOFs with FFC/FIAT 2019 vector-Quadrature component blocks; not runtime MPI',points=len(fields['eF']),max_norm_errors={k:float(np.max(abs(np.linalg.norm(v,axis=1)-1))) for k,v in fields.items()},max_absolute_pairwise_dots={k+'_'+j:float(np.max(abs(np.sum(fields[k]*fields[j],axis=1)))) for k,j in [('eF','eS'),('eF','eN'),('eS','eN')]})
    a.output.mkdir(parents=True,exist_ok=False)
    for name,report in [('candidate_mesh_coordinate_comparison.json',geometry),('candidate_input_basis_check.json',basis)]:
        (a.output/name).write_text(json.dumps(report,indent=2))
    print('candidate coordinate set matches:',geometry['coordinate_set_matches'])


if __name__=='__main__':main()
