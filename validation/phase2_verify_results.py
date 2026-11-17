"""Independent convergence controls and stability diagnostics; no LV solve."""
import argparse
import csv
import json
import math
from pathlib import Path
import runpy
import numpy as np
from FIAT.reference_element import UFCTetrahedron
from FIAT.quadrature_schemes import create_quadrature
from phase2_element_probe import Element,decode


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probe',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--shear',type=Path,required=True);ap.add_argument('--input-audit',type=Path);args=ap.parse_args()
    r=json.loads((args.probe/'report.json').read_text());calc=runpy.run_path(str(args.probe/'generated_source_law_numpy.py'))['calc'];el=Element(np.asarray(r['vertices']),calc);checks=[];refine=[];stiff=[];active=[]
    def check(name,condition,**details):
        checks.append(dict(test=name,passed=bool(condition),details=details))
        if not condition:raise AssertionError(name)
    for degree in (2,4,6,8,10,12):
        q=create_quadrature(UFCTetrahedron(),degree,'default');x=np.asarray(q.get_points());w=np.asarray(q.get_weights());error=0
        for a in range(degree+1):
            for b in range(degree-a+1):
                for c in range(degree-a-b+1):
                    exact=math.factorial(a)*math.factorial(b)*math.factorial(c)/math.factorial(a+b+c+3)
                    value=np.dot(w,x[:,0]**a*x[:,1]**b*x[:,2]**c);error=max(error,abs(value-exact))
        check('FIAT monomial exactness degree %d'%degree,error<1e-12,max_abs_error=error,points=len(w))
    z=el.state(np.eye(3),0);z[30:]=1000
    a=el.assemble(z,2,1000);b=el.assemble(z,10,1000)
    error=np.linalg.norm(a['scaled_K']-b['scaled_K'])/np.linalg.norm(b['scaled_K'])
    check('homogeneous reference constant-pressure K2 vs K10',error<1e-12,relative=error)
    directions=[np.eye(3)[i] for i in range(3)]+[np.array([1.,1,1])/np.sqrt(3),np.array([1.,-1,0])/np.sqrt(2)]
    for name,state in r['states'].items():
        z=np.asarray(state['z']);old=state['old_hsl_nm'];a=el.assemble(z,10,old);b=el.assemble(z,12,old)
        er=np.linalg.norm(a['scaled_R']-b['scaled_R'])/max(np.linalg.norm(b['scaled_R']),1e-14);ek=np.linalg.norm(a['scaled_K']-b['scaled_K'])/np.linalg.norm(b['scaled_K'])
        check('degree10 reference converged vs12 '+name,max(er,ek)<1e-9,R_relative=er,K_relative=ek);refine.append(dict(state=name,R10_to12_relative=er,K10_to12_relative=ek))
        F=np.asarray(state['Fbar']);val=decode(calc(F.ravel(),1000,old,.005));Ap=val[6][0].reshape(3,3,3,3);At=sum(val[6]).reshape(3,3,3,3);low=[];total=[]
        for n in directions:
            normal=np.linalg.inv(F).T.dot(n);_,_,vh=np.linalg.svd(normal.reshape(1,3));B=vh[1:].T
            for A,store in ((Ap,low),(At,total)):
                Q=np.einsum('ijkl,j,l->ik',A,n,n);store.extend(np.linalg.eigvalsh(B.T.dot(Q).dot(B)).tolist())
        eig=np.linalg.eigvalsh(b['scaled_K']);stiff.append(dict(state=name,min_sampled_passive_isochoric_acoustic_Pa=min(low),max_sampled_passive_isochoric_acoustic_Pa=max(low),min_sampled_total_isochoric_acoustic_Pa=min(total),max_sampled_total_isochoric_acoustic_Pa=max(total),mixed_scaled_eigen_min=float(eig.min()),mixed_scaled_eigen_max=float(eig.max()),note='sampled rank-one directions only; mixed eigen signs include pressure saddle point and unconstrained element modes'))
        base=el.assemble(z,8,old,mass=.005);reduced=el.assemble(z,8,old,mass=.0025)
        active.append(dict(state=name,bound_mass_from=.005,bound_mass_to=.0025,scaled_residual_change=float(np.linalg.norm(reduced['scaled_R']-base['scaled_R'])),constraint_residual_change=float(np.linalg.norm(reduced['blocks'][3]-base['blocks'][3])),relative_active_residual_change=float(np.linalg.norm(reduced['blocks'][1]-base['blocks'][1])/max(np.linalg.norm(base['blocks'][1]),1e-14)),interpretation='fixed synthetic state; no resulting displacement was solved'))
        check('pressure constraint unaffected by fixed-state active mass '+name,np.linalg.norm(reduced['blocks'][3]-base['blocks'][3])==0)
    shear=json.loads((args.shear/'report.json').read_text())
    for x in shear['results']:
        err=abs(x['UFL_isochoric_shear_tangent_Pa']-x['finite_difference_tangent_Pa'])/max(abs(x['UFL_isochoric_shear_tangent_Pa']),1.)
        check('UFL acoustic vs FD DOF %d'%x['dof'],err<1e-6,relative=err)
    case=shear['results'][0]
    check('2833 conditional fixed-moment shear tangent negative',case['UFL_isochoric_shear_tangent_Pa']<0)
    for trace in shear['newton_traces']:
        check('reduced Newton convergence init %g damping %g'%(trace['initial'],trace['damping']),trace['converged'],iterations=trace['iterations'],root=trace['root'],curvature=trace['curvature'])
    for state,values in r['UFL_checks'].items():
        for name,value in values.items():
            tolerance=1e-6 if name=='ufl_JVP_vs_FD_relative' else 1e-12
            check('source UFL '+state+' '+name,value<tolerance,relative=value,tolerance=tolerance)
    with (args.probe/'directional_derivatives.csv').open() as f:
        for row in csv.DictReader(f):
            check('nodal FD '+row['state']+' epsilon '+row['epsilon'],float(row['relative_error'])<1e-5,relative=float(row['relative_error']))
    root=args.probe.parent
    with (root/'cavity_probe/convergence.csv').open() as f:
        for row in csv.DictReader(f):
            if int(row['degree'])==4:
                check('cavity degree4 convergence '+row['state'],float(row['K_scaled_relative'])<1e-12,relative=float(row['K_scaled_relative']))
    compiler=json.loads((root/'compiler_audit.json').read_text())
    for form in compiler['forms']:
        check('FFC degree2 metadata '+form['form'],form['status']=='ANALYZED' and all(i['metadata']['quadrature_degree']==2 for i in form.get('integrals',[])))
    input_root=args.input_audit or root
    geometry=json.loads((input_root/'candidate_mesh_coordinate_comparison.json').read_text())
    check('candidate mesh quadrature coordinate set',geometry['coordinate_set_matches'],max_distance=geometry['maximum_nearest_distance'])
    basis=json.loads((input_root/'candidate_input_basis_check.json').read_text())
    check('candidate input basis near-unit/orthogonal',max(list(basis['max_norm_errors'].values())+list(basis['max_absolute_pairwise_dots'].values()))<1e-6)
    rng=np.random.RandomState(17);err=0
    for i in range(200):
        F=np.eye(3)+rng.normal(size=(3,3))*.5;C=F.T.dot(F);lam=np.linalg.norm(F[:,0]);upper=lam*(np.trace(C)-lam*lam)/2;err=max(err,abs(np.linalg.det(F))-upper)
    check('trace/stretch determinant inequality 200 fixtures',err<1e-12,maximum_violation=err)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('stiffness.csv',stiff),('active_perturbations.csv',active),('reference_refinement.csv',refine)]:
        with (args.output/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'tests.json').write_text(json.dumps(dict(tests=checks,passed=all(c['passed'] for c in checks),count=len(checks),evidence_class='isolated numerical controls; not original DOLFIN runtime'),indent=2));print('PASS',len(checks),'focused numerical checks')


if __name__=='__main__':main()
