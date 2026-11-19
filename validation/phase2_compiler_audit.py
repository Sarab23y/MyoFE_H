"""FFC 2019.1 analysis of source-law UFL forms; no DOLFIN assembly."""
import argparse
import json
from pathlib import Path
import numpy as np
import ufl
import ffc
from ffc.analysis import analyze_ufl_objects
from ffc.parameters import default_parameters
from phase2_element_probe import source_law, active_context


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=json.loads((args.repo/'demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json').read_text())
    cell=ufl.tetrahedron;Q=ufl.FiniteElement('Quadrature',cell,2,quad_scheme='default')
    V=ufl.VectorElement('CG',cell,2);P=ufl.FiniteElement('CG',cell,1);Real=ufl.FiniteElement('Real',cell,0)
    W=ufl.MixedElement([V,P,Real,ufl.MixedElement([Real]*5)])
    w=ufl.Coefficient(W);test=ufl.TestFunction(W);trial=ufl.TrialFunction(W)
    u,pressure,pendo,c=ufl.split(w);v,q,qe,vc=ufl.split(test)
    F=ufl.Identity(3)+ufl.grad(u);C=F.T*F;J=ufl.det(F)
    law,cb=source_law(args.repo,cfg,'ufl')
    for group in ('myofiber','ground_matrix','collagen'):
        law.parameters[group]={k:[ufl.Coefficient(Q)] for k in law.parameters[group]}
    for k in ('phi_g','phi_m','phi_c'):law.parameters[k]=[ufl.Coefficient(Q)]
    law.parameters.update(fiber=ufl.Coefficient(ufl.VectorElement('Quadrature',cell,2,quad_scheme='default')),
                          sheet=ufl.Coefficient(ufl.VectorElement('Quadrature',cell,2,quad_scheme='default')),
                          **{'sheet-normal':ufl.Coefficient(ufl.VectorElement('Quadrature',cell,2,quad_scheme='default'))})
    law.parameters['hsl0']=ufl.Coefficient(Q);law.parameters['pressure_variable']=pressure;law.Cmat=lambda:C;law.J=lambda:J
    f0=law.parameters['fiber'];hsl=law.parameters['hsl0']*ufl.sqrt(ufl.inner(f0,C*f0))
    obj=active_context(cfg,ufl.Coefficient(Q));obj.model['functions']={k:ufl.Coefficient(Q) for k in obj.model['functions']}
    old=ufl.Coefficient(Q);S=cb(obj,hsl-old)
    Pactive=S*ufl.outer(f0,f0)
    dx=ufl.dx(domain=w.ufl_domain());ds=ufl.ds(domain=w.ufl_domain());X=ufl.SpatialCoordinate(w.ufl_domain());N=ufl.FacetNormal(w.ufl_domain())
    raw=law._passive_energy_components(C,hsl)
    energy=[law.parameters[k][-1]*r for k,r in zip(('phi_g','phi_m','phi_c'),raw)]
    forms={name:ufl.derivative(E,w,test)*dx for name,E in zip(('ground_R','Xi_R','collagen_R'),energy)}
    forms['pressure_R']=ufl.derivative(-pressure*(J-1),w,test)*dx
    forms['active_R']=ufl.inner(F*Pactive,ufl.grad(v))*dx
    # Exact source cavity integrand; area and prescribed volume represented as
    # scalar constants here because they do not set polynomial integration order.
    V_u=-ufl.inner(ufl.cofac(F)*N,X+u)/3
    forms['cavity_R']=ufl.derivative(pendo*1.0*ds(2)-pendo*V_u*ds(2),w,test)
    rigid=ufl.inner(ufl.as_vector([c[0],c[1],0]),u)+ufl.inner(ufl.as_vector([0,0,c[2]]),ufl.cross(X,u))+ufl.inner(ufl.as_vector([c[3],0,0]),ufl.cross(X,u))+ufl.inner(ufl.as_vector([0,c[4],0]),ufl.cross(X,u))
    forms['rigid_R']=ufl.derivative(rigid*dx,w,test)
    n=J*ufl.inv(F.T)*N
    forms['optional_spring_R']=-.1*ufl.inner(ufl.dot(u,n)*n,v)*ds(1)
    forms['total_R']=sum(forms[k] for k in ('ground_R','Xi_R','collagen_R','pressure_R','active_R','cavity_R','rigid_R'))
    forms['total_Jacobian']=ufl.derivative(forms['total_R'],w,trial)
    result=[]
    for name,form in forms.items():
        params=default_parameters();params['representation']='uflacs';params['quadrature_degree']=2
        try:
            data=analyze_ufl_objects((form,),kind='form',parameters=params)[0][0]
            integrals=[]
            for item in data.integral_data:
                integrals.append(dict(type=item.integral_type,subdomain=str(item.subdomain_id),metadata=item.metadata))
            result.append(dict(form=name,status='ANALYZED',integrals=integrals))
        except Exception as error:result.append(dict(form=name,status='ERROR',error=repr(error)))
        print(name,result[-1]['status'],flush=True)
    args.output.write_text(json.dumps(dict(evidence_class='FFC/UFL analysis with inspected degree-2 parameter override; not original runtime compilation',ffc=ffc.__version__,ufl=ufl.__version__,forms=result),indent=2,default=str))


if __name__=='__main__':main()
