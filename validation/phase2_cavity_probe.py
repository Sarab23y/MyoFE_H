"""Source-equivalent cofactor cavity term on a candidate endocardial face.

Independent local CG2/Real surface assembly; no DOLFIN/global cavity solve.
"""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
import sympy as sy
from FIAT.reference_element import UFCTriangle
from FIAT.quadrature_schemes import create_quadrature
from phase2_element_probe import Element


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mesh',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    with h5py.File(str(args.mesh),'r') as h:
        X=h['ellipsoidal/coordinates'][:];T=h['ellipsoidal/topology'][:];faces=h['ellipsoidal/facetboundaries/topology'][:];marks=h['ellipsoidal/facetboundaries/values'][:].reshape(-1)
    fi=int(np.flatnonzero(marks==2)[0]);face=faces[fi];found=[i for i,c in enumerate(T) if set(face).issubset(c)]
    if len(found)!=1:raise ValueError('Expected one adjacent cell for endocardial boundary')
    cell=found[0];vertices=X[T[cell]];el=Element(vertices,lambda *a:None)
    xf=X[face];cross=np.cross(xf[1]-xf[0],xf[2]-xf[0]);surface_scale=np.linalg.norm(cross);normal=cross/surface_scale
    if np.dot(normal,xf.mean(axis=0)-vertices.mean(axis=0))<0:normal=-normal
    symbols=sy.symbols('z0:12');F=sy.Matrix(3,3,symbols[:9]);x=sy.Matrix(symbols[9:]);N=sy.Matrix(normal.tolist());cofac=F.cofactor_matrix()
    phi=(cofac*N).dot(x)/3;g=sy.Matrix([sy.diff(phi,z) for z in symbols]);H=g.jacobian(symbols)
    fn=sy.lambdify([symbols],[phi,list(g),list(H)],'numpy');records=[];arrays={}
    for name,Fbar,curve in [('reference',np.eye(3),0),('diastolic',np.diag([1.08,1.03,1/(1.08*1.03)]),.025),('systolic',np.diag([.82,1.08,1/(.82*1.08)]),.04),('distorted',np.array([[.35,.35,.1],[0,1.65,.25],[0,0,1/(.35*1.65)]]),.025)]:
        u=el.state(Fbar,curve)[:30].reshape(10,3)
        results={}
        for degree in (2,4,6,8,10):
            q=create_quadrature(UFCTriangle(),degree,'default');t=np.asarray(q.get_points());weights=np.asarray(q.get_weights())*surface_scale
            physical=xf[0]+t[:,0,None]*(xf[1]-xf[0])+t[:,1,None]*(xf[2]-xf[0]);ref=(physical-vertices[0]).dot(el.inv.T);tabs=el.p2.tabulate(1,ref);sh=tabs[(0,0,0)].T;gr=np.stack([tabs[(1,0,0)],tabs[(0,1,0)],tabs[(0,0,1)]],axis=-1).transpose(1,0,2).dot(el.inv)
            R=np.zeros(31);K=np.zeros((31,31));E=0;p=5000.
            for point,weight,Bshape,grad in zip(physical,weights,sh,gr):
                Fq=np.eye(3)+u.T.dot(grad);xq=point+Bshape.dot(u);value,gv,hv=fn(np.r_[Fq.ravel(),xq]);gv=np.asarray(gv);hv=np.asarray(hv).reshape(12,12)
                B=np.zeros((12,30))
                for a in range(10):
                    for i in range(3):B[3*i:3*i+3,3*a+i]=grad[a];B[9+i,3*a+i]=Bshape[a]
                E+=weight*p*value;R[:30]+=weight*p*B.T.dot(gv);R[30]+=weight*value
                K[:30,:30]+=weight*p*B.T.dot(hv).dot(B);K[:30,30]+=weight*B.T.dot(gv);K[30,:30]+=weight*B.T.dot(gv)
            dr=np.r_[np.repeat(1/(1000*el.L**2),30),1/el.L**3];dz=np.r_[np.repeat(el.L,30),1000.]
            results[degree]=(E,R,K,R*dr,dr[:,None]*K*dz[None,:])
        reference=results[10]
        for degree,(E,R,K,Rs,Ks) in results.items():
            records.append(dict(state=name,degree=degree,energy=E,energy_abs_error=abs(E-reference[0]),R_abs_difference=float(np.linalg.norm(R-reference[1])),R_scaled_relative=float(np.linalg.norm(Rs-reference[3])/np.linalg.norm(reference[3])),K_scaled_relative=float(np.linalg.norm(Ks-reference[4])/np.linalg.norm(reference[4]))))
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'convergence.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (args.output/'report.json').write_text(json.dumps(dict(evidence_class='synthetic CG2/Real source-equivalent cavity surface term, not global LV volume constraint',candidate_face=fi,adjacent_candidate_cell=cell,reference_normal=normal.tolist(),prescribed_volume_term=0,pressure_Pa=5000,source='dependencies/forms.py LVV0constrainedE: V_u=-cofac(F)N dot (X+u)/3; Wvol=pendo*V0/area-pendo*V_u',qualification='nonzero prescribed-volume/area term is constant here and integrates exactly; global cavity and rigid constraints not solved'),indent=2))
    print('cavity face',fi,'cell',cell,'degree4 max relative',max(r['K_scaled_relative'] for r in records if r['degree']==4))


if __name__=='__main__':main()
