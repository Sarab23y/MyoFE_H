"""Manufactured vector-CG2 native-hook reachability/invariance probe, not an LV run."""
from __future__ import print_function
import argparse
import os
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    import dolfin as df
    from mpi4py import MPI
    from diagnostic_io import root_call,write_report
    repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0,os.path.join(repo,'python_codes'))
    from LV_simulation.dependencies.native_newton_telemetry import NewtonTelemetry
    from LV_simulation.dependencies.fenics_2017_compat import project_quadrature
    comm=MPI.COMM_WORLD
    mesh=df.UnitCubeMesh(3,3,3)
    V=df.VectorFunctionSpace(mesh,'CG',2)
    Q=df.FunctionSpace(mesh,df.FiniteElement('Quadrature',mesh.ufl_cell(),2,quad_scheme='default'))
    df.parameters['form_compiler']['quadrature_degree']=2
    df.parameters['form_compiler']['representation']='quadrature'
    root_call(comm,lambda:os.makedirs(args.output))
    def solve_once(observe,inject_error=False):
        u=df.interpolate(df.Constant((.5,.2,.1)),V)
        v=df.TestFunction(V)
        target=df.Expression(('2.0+0.1*x[0]','0.1*x[1]','0.1*x[2]'),degree=1)
        # Manufactured reaction-diffusion problem only. The gradient term is
        # integrated exactly at degree two and avoids the known null modes of
        # a pure underintegrated CG2 reaction mass operator. No LV form is used.
        residual=(df.inner(df.grad(u),df.grad(v))+df.inner((1+df.inner(u,u))*u-target,v))*df.dx
        jac=df.derivative(residual,u)
        monitor=None
        if observe:
            previous=[None]
            def sink(kind,values):
                injected = inject_error and comm.Get_rank()==comm.Get_size()-1
                if any(comm.allgather(injected)):
                    raise RuntimeError('intentional observer-only failure fixture on last rank')
                current=u.vector().get_local().copy()
                if previous[0] is not None:
                    difference=current-previous[0]
                    parts=comm.allgather((float(np.dot(difference,difference)),float(np.max(np.abs(difference)))))
                    norm=np.sqrt(sum(x[0] for x in parts));maximum=max(x[1] for x in parts)
                    if not np.isclose(norm,values['correction']['applied_displacement_l2'],atol=1.e-12,rtol=1.e-10):
                        raise RuntimeError('native displacement correction norm/stage mismatch')
                    if not np.isclose(maximum,values['correction']['applied_displacement_max'],atol=1.e-12,rtol=1.e-10):
                        raise RuntimeError('native maximum correction mismatch')
                previous[0]=current
            def metrics():
                F=df.Identity(3)+df.grad(u)
                fields=np.column_stack([project_quadrature(F[i,j],Q,'native_probe_F%d%d'%(i,j),comm).vector().get_local()
                    for i in range(3) for j in range(3)]).reshape((-1,3,3))
                J=np.linalg.det(fields);stretch=np.linalg.norm(fields[:,:,0],axis=1)
                parts=comm.allgather((float(np.min(J)),float(np.max(J)),float(np.min(stretch))))
                return dict(J_min=min(x[0] for x in parts),J_max=max(x[1] for x in parts),stretch_min=min(x[2] for x in parts))
            monitor=NewtonTelemetry(lambda:u.vector().get_local(),sink,comm,metrics)
            monitor.start()
        try:
            # Same explicit NSolver tolerances; same original DOLFIN solve call in both arms.
            df.solve(residual==0,u,J=jac,solver_parameters={'newton_solver':{
                'relative_tolerance':1.e-7,'absolute_tolerance':1.e-7,'maximum_iterations':50}},
                form_compiler_parameters={'representation':'uflacs'})
            try:result=monitor.finish(True) if monitor else None
            except RuntimeError:
                if not inject_error:raise
                result=dict(expected_observer_error_caught=True)
            if inject_error and not result.get('expected_observer_error_caught'):
                raise RuntimeError('observer error fixture did not fire')
        finally:
            if monitor:monitor.close()
        return u.vector().get_local().copy(),result
    before,_=solve_once(False)
    after,telemetry=solve_once(True)
    failed_observer,failure_result=solve_once(True,True)
    difference=max(comm.allgather(float(np.max(np.abs(before-after)))))
    if not all(comm.allgather(bool(np.array_equal(before,after)))):
        raise RuntimeError('native observer changed manufactured solution: %.17g'%difference)
    if not all(comm.allgather(bool(np.array_equal(before,failed_observer)))):
        raise RuntimeError('observer error altered original manufactured Newton solution')
    initial=telemetry['records'][0]['trial']
    if not all(abs(initial[key]-1.)<1.e-10 for key in ('J_min','J_max','stretch_min')):
        raise RuntimeError('initial constant displacement must give F=I, J=1, stretch=1')
    root_call(comm,lambda:write_report(os.path.join(args.output,'native_probe.json'),dict(
        passed=True,dolfin_version=df.__version__,mpi_size=comm.Get_size(),max_coefficient_difference=difference,
        telemetry_availability="residual, correction and trial metrics recorded by reachable native hooks",
        exact_solution_equal=True,telemetry=telemetry,
        one_rank_observer_failure_deferred_and_solution_unchanged=True,
        comparison='same loaded forwarder, callback inactive versus active',
        pure_process_baseline_still_required=True,manufactured_vector_problem=True,
        LV_invariance_validated=False)))


if __name__=='__main__':main()
