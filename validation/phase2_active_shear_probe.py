"""Source-UFL fixed-moment shear tests, NOT a historical or global FE solve."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import ufl
from ufl.algorithms import expand_derivatives
from scipy.optimize import brentq
from phase2_element_probe import source_law, active_context


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--event-csv',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads((args.repo/'demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json').read_text())
    import pandas as pd
    allrows=pd.read_csv(str(args.event_csv));events={2833:8.696,4844:20.273,1580:25.165,2497:35.165};rows=[];curves=[];traces=[]
    for gid,time in events.items():
        d=allrows[allrows.dof==gid];r=d.iloc[np.argmin(abs(d.time.values-time))];prev=d.iloc[np.argmin(abs(d.time.values-(time-.001)))];lam=float(r.fiber_stretch);h0=float(r.HSL/lam);mass=float(r.n_bound);old=float(prev.HSL)
        myo=cfg['model']['half_sarcomere']['myofilaments'];rho=myo['cb_number_density'][0];k=myo['k_cb'][0];c=myo['implementation']['filament_compliance_factor'][0];xps=myo['x_ps'][0];D=rho*k*1e-9
        mean=float(r.active_saved)/(D*mass)-xps
        obj=active_context(cfg,mass);obj.y_split[:]=0;grid=obj.hs.myof.x
        if not grid[0]<=mean<=grid[-1]:raise ValueError('Reconstructed first moment outside candidate bins')
        hi=min(max(int(np.searchsorted(grid,mean)),1),len(grid)-1);lo=hi-1;weight=(mean-grid[lo])/(grid[hi]-grid[lo]);obj.y_split[2+lo]=mass*(1-weight);obj.y_split[2+hi]=mass*weight
        # This two-bin vector only represents measured aggregate moments under
        # stated assumptions. It is explicitly NOT the historical population.
        fc=ufl.Coefficient(ufl.TensorElement('Real',ufl.tetrahedron,0,shape=(3,3)));F=ufl.variable(fc);C=F.T*F
        law,cb=source_law(args.repo,cfg,'ufl');law.Cmat=lambda:C;law.J=lambda:ufl.det(F);law.parameters['hsl0']=h0;law.parameters['pressure_variable']=1000.
        stretch=ufl.sqrt(C[0,0]);S=cb(obj,h0*stretch-old);W=law.PassiveMatSEF(h0*stretch)
        P=expand_derivatives(ufl.diff(W,F)+F*ufl.as_matrix([[S,0,0],[0,0,0],[0,0,0]]));A=expand_derivatives(ufl.diff(P,F))
        Fbase=np.diag([lam,1/np.sqrt(lam),1/np.sqrt(lam)])
        mapping={fc:Fbase};position=(.2,.2,.2)
        q=float(A[1,0,1,0](position,mapping));g=law.parameters['phi_g'][-1]*law.parameters['ground_matrix']['a_g'][-1]*np.exp(law.parameters['ground_matrix']['b_g'][-1]*(np.trace(Fbase.T.dot(Fbase))-3))
        pp=expand_derivatives(ufl.diff(law._weighted_passive_energy(C,h0*stretch),F))
        passive_shear=float(pp[0,0](position,mapping))/lam
        eq=float(S(position,mapping));expected=passive_shear+eq
        eps=1e-6;fp=Fbase.copy();fm=Fbase.copy();fp[1,0]+=eps;fm[1,0]-=eps
        fd=(P[1,0](position,{fc:fp})-P[1,0](position,{fc:fm}))/(2*eps)
        b=D*mass*c*h0;a=D*mass*(mean+xps-c*old);bg=law.parameters['ground_matrix']['b_g'][-1]
        # On this rank-one isochoric path fiber remains compressed near zero;
        # sheet/normal collagen invariants are unchanged. Active primitive is
        # only a fixed-population mathematical potential, not MyoSim free energy.
        def residual(gamma):
            l=np.sqrt(lam*lam+gamma*gamma)
            return gamma*(g*np.exp(bg*gamma*gamma)+a+b*l)
        def tangent(gamma):
            l=np.sqrt(lam*lam+gamma*gamma);G=g*np.exp(bg*gamma*gamma)
            return G+a+b*l+gamma*gamma*(2*bg*G+b/l)
        roots=[0.]
        if q<0:
            root=brentq(lambda x:residual(x)/x,1e-9,min(.8,np.sqrt(1-lam*lam)*.99));roots +=[-root,root]
        for init in ([0.,-.1,.1] if gid==2833 else []):
            for damping in (1.,.5):
                gamma=init;history=[]
                for it in range(80):
                    value=residual(gamma);correction=0. if abs(value)<1e-9 else -damping*value/tangent(gamma)
                    history.append(dict(iteration=it,gamma=gamma,residual_Pa=value,correction=correction,J=1.,minimum_singular_value=float(np.linalg.svd(Fbase+np.array([[0,0,0],[gamma,0,0],[0,0,0]]))[1][-1])))
                    if abs(value)<1e-9:break
                    gamma+=correction
                traces.append(dict(dof=gid,initial=init,damping=damping,converged=bool(abs(history[-1]['residual_Pa'])<1e-9),root=gamma,curvature=tangent(gamma),iterations=len(history)-1,history=history))
        row=dict(dof=gid,evidence_class='conditional moment reconstruction plus synthetic isochoric geometry; NOT historical populations or equilibrium',lambda_f=lam,hsl0=h0,mass=mass,mean_x_nm=mean,pre_reset_active_Pa=eq,saved_post_reset_Pa=float(r.active_saved),ground_shear_Pa=float(g),passive_total_shear_Pa=passive_shear,UFL_isochoric_shear_tangent_Pa=q,analytic_tangent_Pa=float(expected),finite_difference_tangent_Pa=float(fd),J=1.,reduced_shear_roots=roots,reduced_equilibrium_probed=gid==2833)
        rows.append(row)
        for gamma in np.linspace(-.4,.4,401):
            if gid!=2833:continue
            if lam<1 and np.sqrt(lam*lam+gamma*gamma)>=1:continue
            curves.append(dict(dof=gid,gamma=gamma,projected_shear_residual_Pa=float(residual(gamma))))
        print(gid,'Sactive',eq,'ground shear',g,'total tangent',q,'roots',roots,flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(dict(evidence_class='isolated source-UFL constitutive/acoustic and reduced-mode tests; no DOLFIN solve',assumptions=['unit orthonormal synthetic basis','candidate material/rate coefficients','archived scalar stage identities and point matching','synthetic isochoric diagonal F, not reconstructed original F','two-bin moment representative is NOT historical y'],results=rows,newton_traces=traces),indent=2))
    with (args.output/'shear_curves.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)


if __name__=='__main__':main()
