"""Independent CG2/CG1 local integration using actual source law bodies.

No DOLFIN assembly or LV solve. Constant/smooth synthetic coefficients extend
material-point data to higher rules explicitly; historical data are not inferred.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap
import types

import h5py
import numpy as np
import sympy as sy
from sympy.printing.pycode import NumPyPrinter
from FIAT.reference_element import UFCTetrahedron
from FIAT.lagrange import Lagrange
from FIAT.quadrature_schemes import create_quadrature


def extract(path, name):
    text=Path(path).read_text()
    start=text.index('    def '+name+'(')
    end=text.find('\n    def ', start+5)
    return textwrap.dedent(text[start:end if end>=0 else len(text)])


def source_law(repo, cfg, backend):
    if backend=='sympy':
        scope={'np':np, 'exp':sy.exp,'tr':sy.trace,
               'inner':lambda a,b:(a.T*b)[0],
               'conditional':lambda c,a,b:sy.Piecewise((a,c),(b,True))}
    else:
        import ufl
        scope=dict(vars(ufl));scope['np']=np
    for name in ('_passive_energy_components','_weighted_passive_energy','PassiveMatSEF'):
        exec(compile(extract(repo/'python_codes/LV_simulation/dependencies/forms.py',name),'<source '+name+'>','exec'),scope)
    exec(compile(extract(repo/'python_codes/LV_simulation/mesh/mesh.py','return_cb_stress'),'<source FE CB>','exec'),scope)
    class Law:
        pass
    for name in ('_passive_energy_components','_weighted_passive_energy','PassiveMatSEF'):
        setattr(Law,name,scope[name])
    law=Law();law.parameters=dict(cfg['mesh']['forms_parameters']['passive_law_parameters'])
    law.parameters['hsl0']=1000.;law.parameters['incompressible']=True
    if backend=='sympy':
        law.parameters.update(fiber=sy.Matrix([1,0,0]),sheet=sy.Matrix([0,1,0]),**{'sheet-normal':sy.Matrix([0,0,1])})
    else:
        law.parameters.update(fiber=ufl.as_vector([1,0,0]),sheet=ufl.as_vector([0,1,0]),**{'sheet-normal':ufl.as_vector([0,0,1])})
    return law,scope['return_cb_stress']


def active_context(cfg,mass):
    myo=cfg['model']['half_sarcomere']['myofilaments']
    impl={k:v[0] for k,v in myo['implementation'].items()}
    if impl['kinetic_scheme']!='3_state_with_SRX':raise ValueError('Explicit three-state fixture only')
    x=np.arange(impl['bin_min'],impl['bin_max']+impl['bin_width'],impl['bin_width'])
    obj=types.SimpleNamespace(hs=types.SimpleNamespace(myof=types.SimpleNamespace(implementation=impl,x=x,no_of_x_bins=len(x))))
    obj.y_split=np.array([0,0]+[mass/len(x)]*len(x)+[0,0],dtype=object)
    obj.model={'functions':{k:myo[k][0] for k in ('cb_number_density','k_cb','x_ps')}}
    return obj


def constitutive(repo,cfg,out):
    f=sy.symbols('f0:9',real=True);p,old,m=sy.symbols('p old mass',real=True)
    F=sy.Matrix(3,3,f);C=F.T*F;J=F.det();lam=sy.sqrt(C[0,0])
    law,cb=source_law(repo,cfg,'sympy');law.Cmat=lambda:C;law.J=lambda:J;law.parameters['pressure_variable']=p
    raw=law._passive_energy_components(C,1000*lam)
    frac=[law.parameters[k][-1] for k in ('phi_g','phi_m','phi_c')]
    energies=[r*w for r,w in zip(raw,frac)]
    Wpass=sum(energies);Wp=law.PassiveMatSEF(1000*lam)-Wpass
    Ppass=sy.Matrix([sy.diff(Wpass,z) for z in f]);Ppress=sy.Matrix([sy.diff(Wp,z) for z in f])
    s=cb(active_context(cfg,m),1000*lam-old)
    Pact=sy.Matrix([F[i,0]*s if j==0 else 0 for i in range(3) for j in range(3)])
    Ps=[Ppass,Pact,Ppress];As=[q.jacobian(f) for q in Ps]
    expr=energies+[Wp,J,lam,s]+[z for q in Ps for z in q]+[z for a in As for z in a]+[sy.diff(J,z) for z in f]
    replacements,reduced=sy.cse(expr,symbols=sy.numbered_symbols('v'))
    printer=NumPyPrinter();lines=['def calc(F,p,old,mass):','    '+','.join(str(z) for z in f)+' = F']
    lines += ['    %s = %s'%(a,printer.doprint(b)) for a,b in replacements]
    lines += ['    return numpy.array(['+','.join(printer.doprint(z) for z in reduced)+'], dtype=float)']
    code='\n'.join(lines);(out/'generated_source_law_numpy.py').write_text('import numpy\n'+code+'\n')
    scope={'numpy':np};exec(compile(code,'<CSE source-law derivative>','exec'),scope)
    return scope['calc']


def decode(v):
    return v[:3],v[3],v[4],v[5],v[6],[v[7+i*9:16+i*9].reshape(3,3) for i in range(3)], [v[34+i*81:115+i*81].reshape(9,9) for i in range(3)],v[-9:].reshape(3,3)


class Element:
    def __init__(self,vertices,calc):
        self.vertices=vertices;self.B=(vertices[1:]-vertices[0]).T
        self.inv=np.linalg.inv(self.B);self.det=abs(np.linalg.det(self.B));self.L=self.det**(1/3)
        self.p2=Lagrange(UFCTetrahedron(),2);self.p1=Lagrange(UFCTetrahedron(),1);self.calc=calc
        self.nodes=np.array([list(n.get_point_dict())[0] for n in self.p2.dual.nodes]).dot(self.B.T)+vertices[0]
        self.pnodes=np.array([list(n.get_point_dict())[0] for n in self.p1.dual.nodes]).dot(self.B.T)+vertices[0]
        self.center=vertices.mean(axis=0);self.scale=1000.
        self.rules={}
        for degree in (2,4,6,8,10,12):
            rule=create_quadrature(UFCTetrahedron(),degree,'default');q=np.asarray(rule.get_points());w=np.asarray(rule.get_weights())*self.det
            tab=self.p2.tabulate(1,q);grad=np.stack([tab[(1,0,0)],tab[(0,1,0)],tab[(0,0,1)]],axis=-1).transpose(1,0,2).dot(self.inv)
            shape=self.p1.tabulate(0,q)[(0,0,0)].T
            self.rules[degree]=(q,w,grad,shape)

    def state(self,Fbar,curve):
        x=self.nodes-self.center
        u=x.dot((Fbar-np.eye(3)).T)+.5*curve/self.L*x*x
        p=1000+100*np.sum((self.pnodes-self.center)/self.L,axis=1)
        return np.r_[u.ravel(),p]

    def assemble(self,z,degree,old,mass=.005,tangent=True):
        u=z[:30].reshape(10,3);p=z[30:];q,w,grad,shape=self.rules[degree]
        residuals=np.zeros((4,34));K=np.zeros((34,34));ener=np.zeros(4);jmin=np.inf;svmin=np.inf;stretch=[]
        for weight,g,N in zip(w,grad,shape):
            F=np.eye(3)+u.T.dot(g);vals=self.calc(F.ravel(),float(N.dot(p)),old,mass)
            W,Wp,J,lam,s,Ps,As,DJ=decode(vals)
            if not np.isfinite(vals).all():raise FloatingPointError('Nonfinite constitutive fixture')
            ener+=weight*np.r_[W,Wp];jmin=min(jmin,J);svmin=min(svmin,np.linalg.svd(F)[1][-1]);stretch.append(lam)
            # B maps node-major displacement increments to row-major delta F.
            B=np.zeros((9,30))
            for a in range(10):
                for i in range(3):B[3*i:3*i+3,3*a+i]=g[a]
            for k in range(3):residuals[k,:30]+=weight*B.T.dot(Ps[k].ravel())
            residuals[3,30:]-=weight*N*(J-1)
            if tangent:
                K[:30,:30]+=weight*B.T.dot(sum(As)).dot(B)
                cross=-weight*np.outer(B.T.dot(DJ.ravel()),N)
                K[:30,30:]+=cross;K[30:,:30]+=cross.T
        dr=np.r_[np.repeat(1/(self.scale*self.L**2),30),np.repeat(1/self.L**3,4)]
        dz=np.r_[np.repeat(self.L,30),np.repeat(self.scale,4)]
        return dict(R=residuals.sum(axis=0),blocks=residuals,K=K,energies=ener,Jmin=jmin,smin=svmin,
                    scaled_R=residuals.sum(axis=0)*dr,scaled_K=dr[:,None]*K*dz[None,:],dr=dr,dz=dz,lambda_range=[min(stretch),max(stretch)])


def ufl_checks(repo,cfg,F,p,old,mass,direction,calc):
    import ufl
    from ufl.algorithms import expand_derivatives
    fc=ufl.Coefficient(ufl.TensorElement('Real',ufl.tetrahedron,0,shape=(3,3)))
    pc=ufl.Coefficient(ufl.FiniteElement('Real',ufl.tetrahedron,0));Fv=ufl.variable(fc)
    law,cb=source_law(repo,cfg,'ufl');C=Fv.T*Fv;J=ufl.det(Fv);law.Cmat=lambda:C;law.J=lambda:J;law.parameters['pressure_variable']=pc
    lam=ufl.sqrt(C[0,0]);W=law.PassiveMatSEF(1000*lam)
    S=cb(active_context(cfg,mass),1000*lam-old)
    Pa=Fv*ufl.as_matrix([[S,0,0],[0,0,0],[0,0,0]])
    P=expand_derivatives(ufl.diff(W,Fv)+Pa);A=expand_derivatives(ufl.diff(P,Fv))
    mapping={fc:F,pc:p};x=(.2,.2,.2)
    pp=np.array([[P[i,j](x,mapping) for j in range(3)] for i in range(3)])
    hv=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):hv[i,j]+=A[i,j,k,l](x,mapping)*direction[k,l]
    vals=decode(calc(F.ravel(),p,old,mass));pnp=sum(vals[5]);hnp=sum(vals[6]).dot(direction.ravel()).reshape(3,3)
    eps=1e-6
    plus=sum(decode(calc((F+eps*direction).ravel(),p,old,mass))[5]);minus=sum(decode(calc((F-eps*direction).ravel(),p,old,mass))[5]);fd=(plus-minus)/(2*eps)
    return dict(ufl_vs_independent_P_relative=np.linalg.norm(pp-pnp)/max(np.linalg.norm(pp),1.),
                ufl_vs_independent_JVP_relative=np.linalg.norm(hv-hnp)/max(np.linalg.norm(hv),1.),
                ufl_JVP_vs_FD_relative=np.linalg.norm(hv-fd)/max(np.linalg.norm(hv),1.))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--mesh',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((args.repo/'demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json').read_text())
    calc=constitutive(args.repo,cfg,out)
    with h5py.File(str(args.mesh),'r') as h:xyz=h['ellipsoidal/coordinates'][:];top=h['ellipsoidal/topology'][:]
    volumes=np.abs(np.linalg.det((xyz[top[:,1:]]-xyz[top[:,0,None]]).transpose(0,2,1)))/6
    cell=int(np.argsort(volumes)[len(volumes)//2]);el=Element(xyz[top[cell]],calc)
    states=[('reference',np.eye(3),0),('diastolic_proxy',np.diag([1.08,1.03,1/(1.08*1.03)]),.025),('systolic_proxy',np.diag([.82,1.08,1/(.82*1.08)]),.04),('compressed',np.diag([.55,1.24,1/(.55*1.24)]),.04),('distorted',np.array([[.35,.35,.10],[0,1.65,.25],[0,0,1/(.35*1.65)]]),.025)]
    rows=[];details={};fdrows=[];checks={};rng=np.random.RandomState(14)
    for name,Fbar,curve in states:
        z=el.state(Fbar,curve);old=1000*np.linalg.norm(Fbar[:,0]);runs={d:el.assemble(z,d,old) for d in (2,4,6,8,10)};ref=runs[10]
        v=rng.normal(size=34);v/=np.linalg.norm(v)
        for d,r in runs.items():
            row=dict(state=name,degree=d,points=len(el.rules[d][0]),Jmin=r['Jmin'],minimum_singular_value=r['smin'],
                     passive_energy=sum(r['energies'][:3]),ground_energy=r['energies'][0],Xi_energy=r['energies'][1],collagen_energy=r['energies'][2],pressure_energy=r['energies'][3],
                     R_absolute=np.linalg.norm(r['R']),R_difference_absolute=np.linalg.norm(r['R']-ref['R']),
                     R_scaled_relative=np.linalg.norm(r['scaled_R']-ref['scaled_R'])/max(np.linalg.norm(ref['scaled_R']),1e-14),
                     K_scaled_relative=np.linalg.norm(r['scaled_K']-ref['scaled_K'])/max(np.linalg.norm(ref['scaled_K']),1e-14),
                     JVP_scaled_relative=np.linalg.norm((r['scaled_K']-ref['scaled_K']).dot(v))/max(np.linalg.norm(ref['scaled_K'].dot(v)),1e-14))
            for k,label in enumerate(('passive','active','pressure_u','constraint')):
                row[label+'_R_difference_absolute']=np.linalg.norm(r['blocks'][k]-ref['blocks'][k])
                row[label+'_R_relative']=np.linalg.norm(r['blocks'][k]-ref['blocks'][k])/max(np.linalg.norm(ref['blocks'][k]),1e-12)
            row['active_virtual_work']=float(r['blocks'][1].dot(v*ref['dz']))
            rows.append(row)
        details[name]=dict(z=z.tolist(),old_hsl_nm=old,Fbar=Fbar.tolist(),curvature=curve,quadrature10_lambda_range=ref['lambda_range'])
        if name!='reference':
            for eps in (1e-4,1e-5,1e-6):
                plus=el.assemble(z+eps*v*ref['dz'],8,old,tangent=False)['scaled_R'];minus=el.assemble(z-eps*v*ref['dz'],8,old,tangent=False)['scaled_R'];fd=(plus-minus)/(2*eps);jv=runs[8]['scaled_K'].dot(v)
                fdrows.append(dict(state=name,epsilon=eps,relative_error=np.linalg.norm(jv-fd)/max(np.linalg.norm(jv),1e-14)))
            print('UFL check',name,flush=True);checks[name]=ufl_checks(args.repo,cfg,Fbar,1000,old,.005,rng.normal(size=(3,3))*.01,calc)
        print('state complete',name,flush=True)
    def csvwrite(path,records):
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    csvwrite(out/'integration.csv',rows);csvwrite(out/'directional_derivatives.csv',fdrows)
    report=dict(evidence_class='synthetic local CG2/CG1 assembly, actual source law bodies; not DOLFIN/global LV solve',mesh_path=str(args.mesh),mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),candidate_cell=cell,vertices=el.vertices.tolist(),mesh_cells=len(top),mesh_vertices=len(xyz),states=details,UFL_checks=checks,material_coefficients='constant candidate configuration; unit synthetic fiber basis; not reconstructed historical fields',higher_degree_extension='constant coefficients explicitly defined at every point; no interpolation of historical Quadrature state',boundary_conditions='none for local integration comparison; identical local FIAT node-major ordering at every degree',reference_degree=10)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    # Full numerical vectors/matrices permit independent examination.
    np.savez_compressed(str(out/'local_reference_arrays.npz'),R=ref['R'],K=ref['K'],scaled_K=ref['scaled_K'])


if __name__=='__main__':main()
