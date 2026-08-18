# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:06:26 2019
@author: ani228
"""

from dolfin import *
import sys
import numpy as np

class Forms(object):

    def __init__(self, params):    # amir: sth like constructor

        self.parameters = self.default_parameters()
        self.parameters.update(params)
        if "Fg" in self.parameters:
            self.Fg = self.parameters["Fg"]
            self.M1ij = self.parameters["M1ij"]
            self.M2ij = self.parameters["M2ij"]
            self.M3ij = self.parameters["M3ij"]
            #self.TF = self.parameters["TF"]
        else:
            self.Fg = Identity(3)

    def default_parameters(self):
        return {#"bff"  : 29.0,
			#"bfx"  : 13.3,
			#"bxx"  : 26.6,
			"Kappa": 1e5,
			"incompressible" : True,
			};


    def Fmat(self):

        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        return F

    def update_Fg(self,theta1,theta2,theta3):
        Fg = self.Fg
        M1ij = self.M1ij
        M2ij = self.M2ij
        M3ij = self.M3ij
        #Fg = theta1*M1ij + theta2*M2ij + theta3*M3ij
        temp_Fg = theta1*M1ij + theta2*M2ij + theta3*M3ij
        Fg = project(temp_Fg,self.parameters["growth_tensor_FS"],
                    form_compiler_parameters={"representation":"uflacs"})
        #print "Fg updated", project(Fg,self.TF).vector().get_local()
        self.Fg = Fg

    def Fe(self):
        #Fg = self.parameters["growth_tensor"]
        F = self.Fmat()

        #Fg = self.Fg
        #Fe = as_tensor(F[i,j]*inv(Fg)[j,k], (i,k))
        if "Fg" in self.parameters:
            Fg = self.Fg
            #Fe = F* inv(Fg)
            Fe = as_tensor(F[i,j]*inv(Fg)[j,k], (i,k))
        else:
            Fe = F
        return Fe

    def Emat(self):

        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        #F = self.Fmat()
        F = self.Fe()
        #return 0.5*(F.T*F-I)
    	return 0.5*(as_tensor(F[k,i]*F[k,j] - I[i,j], (i,j)))


    def Cmat(self):

        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        #F = self.Fmat()
        F = self.Fe()
        return F.T*F
        #return as_tensor(F[k,i]*F[k,j],(i,j))

    def J(self):
        #F = self.Fmat()
        F = self.Fe()
        return det(F)


    def LVcavityvol(self):

        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(subdomain_data = self.parameters["facetboundaries"])

        F = self.Fmat()
        #F = self.Fe()

        vol_form = -Constant(1.0/3.0) * inner(det(F)*dot(inv(F).T, N), X + u)*ds(self.parameters["LVendoid"])

        return assemble(vol_form, form_compiler_parameters={"representation":"uflacs"})
    
    def LVV0constrainedE(self):


        mesh = self.parameters["mesh"]
        u = self.parameters["displacement_variable"]
        ds = dolfin.ds(subdomain_data = self.parameters["facetboundaries"])
        dsendo = ds(self.parameters["LVendoid"], domain = self.parameters["mesh"], subdomain_data = self.parameters["facetboundaries"])
        pendo = self.parameters["lv_volconst_variable"]
        V0= self.parameters["lv_constrained_vol"]

        X = SpatialCoordinate(mesh)
        x = u + X

        F = self.Fmat()
        #F = self.Fe()

        N = self.parameters["facet_normal"]
        n = cofac(F)*N

        #n = det(F)*dot(inv(F).T, N)
        #vol_form = -Constant(1.0/3.0) * inner(det(F)*dot(inv(F).T, N), X + u)*ds(self.parameters["LVendoid"])

        area = assemble(Constant(1.0) * dsendo, form_compiler_parameters={"representation":"uflacs"})
        V_u = - Constant(1.0/3.0) * inner(n, x)
        #Wvol = (Constant(1.0/area) * pendo  * V0 * dsendo) - (pendo * V_u *dsendo)
        Wvol = (Constant(1.0/area) * pendo  * V0 * ds(self.parameters["LVendoid"])) - (pendo * V_u *ds(self.parameters["LVendoid"]))

        return Wvol

    def RVcavityvol(self):

        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(subdomain_data = self.parameters["facetboundaries"])

        F = self.Fmat()

        vol_form = -Constant(1.0/3.0) * inner(det(F)*dot(inv(F).T, N), X + u)*ds(self.parameters["RVendoid"])

        return assemble(vol_form, form_compiler_parameters={"representation":"uflacs"})


    def LVcavitypressure(self):

        W = self.parameters["mixedfunctionspace"]
        w = self.parameters["mixedfunction"]
        mesh = self.parameters["mesh"]

        comm = W.mesh().mpi_comm()
        dofmap =  W.sub(self.parameters["LVendo_comp"]).dofmap()
        val_dof = dofmap.cell_dofs(0)[0]

	    # the owner of the dof broadcasts the value
        own_range = dofmap.ownership_range()

        try:
            val_local = w.vector()[val_dof][0]
        except IndexError:
                val_local = 0.0


        pressure = MPI.sum(comm, val_local)

        return pressure



    def RVcavitypressure(self):

        W = self.parameters["mixedfunctionspace"]
        w = self.parameters["mixedfunction"]
        mesh = self.parameters["mesh"]

        comm = W.mesh().mpi_comm()
        dofmap =  W.sub(self.parameters["RVendo_comp"]).dofmap()
        val_dof = dofmap.cell_dofs(0)[0]

	    # the owner of the dof broadcasts the value
        own_range = dofmap.ownership_range()

        try:
            val_local = w.vector()[val_dof][0]
        except IndexError:
            val_local = 0.0


        pressure = MPI.sum(comm, val_local)

        return pressure

    def TempActiveStress(self,time):

        f0 = self.parameters["fiber"]
        #cbforce = Expression('A*(B+sin((B/C)*time + D))', A=30000., B=1., C=16., D=80.2, time = time, degree=0)
        cbforce = Expression(("f"), f=0, degree=1)
        Pactive = cbforce * as_tensor(f0[i]*f0[j], (i,j))
        return Pactive, cbforce

    def _passive_energy_components(self, Cmat):
        """Return the three unweighted Holzapfel--Ogden energies."""
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        a = self.parameters["a"][-1]
        b = self.parameters["b"][-1]
        a_f = self.parameters["a_f"][-1]
        b_f = self.parameters["b_f"][-1]
        a_s = self.parameters["a_s"][-1]
        b_s = self.parameters["b_s"][-1]
        a_fs = self.parameters["a_fs"][-1]
        b_fs = self.parameters["b_fs"][-1]

        I1 = tr(Cmat)
        I4f = inner(f0, Cmat*f0)
        I4s = inner(s0, Cmat*s0)
        I8fs = inner(f0, Cmat*s0)

        # The former passive fiber law was tension-only. Apply that convention
        # to the HO directional terms, but not to the isotropic or shear terms.
        I4f_eff = conditional(I4f > 1.0, I4f, 1.0)
        I4s_eff = conditional(I4s > 1.0, I4s, 1.0)

        W_bulk = a/(2.0*b)*(exp(b*(I1 - 3.0)) - 1.0)
        W_myo = a_f/(2.0*b_f)*(exp(b_f*(I4f_eff - 1.0)**2.0) - 1.0)
        # s0 is the collagen-associated anisotropic direction until a distinct
        # collagen orientation field is introduced.
        W_collagen = \
            a_s/(2.0*b_s)*(exp(b_s*(I4s_eff - 1.0)**2.0) - 1.0) + \
            a_fs/(2.0*b_fs)*(exp(b_fs*I8fs**2.0) - 1.0)
        return W_bulk, W_myo, W_collagen

    def _weighted_passive_energy(self, Cmat):
        W_bulk, W_myo, W_collagen = \
            self._passive_energy_components(Cmat)
        # These dimensionless fields are currently constituent mixture
        # fractions; no density conversion is applied.
        phi_m = self.parameters["phi_m"][-1]
        phi_c = self.parameters["phi_c"][-1]
        phi_g = self.parameters["phi_g"][-1]
        return phi_g*W_bulk + phi_m*W_myo + phi_c*W_collagen

    def PassiveMatSEF(self,hsl):
        # hsl remains in the public signature for callers and active mechanics,
        # although the passive HO response depends on the elastic C tensor.
        Wp = self._weighted_passive_energy(self.Cmat())
        if self.parameters["incompressible"]:
            p = self.parameters["pressure_variable"]
            return Wp - p*(self.J() - 1.0)
        return Wp + self.parameters["Kappa"]/2.0*(self.J() - 1.0)**2.0

    def PassiveMatSEFComps(self,hsl):
        W_bulk, W_myo, W_collagen = \
            self._passive_energy_components(self.Cmat())
        phi_m = self.parameters["phi_m"][-1]
        phi_c = self.parameters["phi_c"][-1]
        phi_g = self.parameters["phi_g"][-1]
        W_myo_weighted = phi_m*W_myo
        W_matrix_weighted = phi_g*W_bulk + phi_c*W_collagen
        if self.parameters["incompressible"]:
            W_matrix_weighted -= \
                self.parameters["pressure_variable"]*(self.J() - 1.0)
        else:
            W_matrix_weighted += \
                self.parameters["Kappa"]/2.0*(self.J() - 1.0)**2.0
        return W_myo_weighted, W_matrix_weighted


    def RVV0constrainedE(self):


        mesh = self.parameters["mesh"]
        self.parameters["displacement_variable"]
        ds = dolfin.ds(subdomain_data = self.parameters["facetboundaries"])
        dsendo = ds(self.parameters["RVendoid"], domain = self.parameters["mesh"], subdomain_data = self.parameters["facetboundaries"])
        pendo = self.parameters["rv_volconst_variable"]
        V0= self.parameters["rv_constrained_vol"]

        X = SpatialCoordinate(mesh)
        x = u + X

        F = self.Fmat()
        N = self.parameters["facet_normal"]
        n = cofac(F)*N

        area = assemble(Constant(1.0) * dsendo, form_compiler_parameters={"representation":"uflacs"})
        V_u = - Constant(1.0/3.0) * inner(x, n)
        Wvol = (Constant(1.0/area) * pendo  * V0 * dsendo) - (pendo * V_u *dsendo)

        return Wvol


    def _passive_stress_components(self):
        """Derive weighted constituent PK2 stresses from the HO energies."""
        Ctensor = variable(self.Cmat())
        W_bulk, W_myo, W_collagen = \
            self._passive_energy_components(Ctensor)
        phi_m = self.parameters["phi_m"][-1]
        phi_c = self.parameters["phi_c"][-1]
        phi_g = self.parameters["phi_g"][-1]

        bulk_passive = 2.0*diff(phi_g*W_bulk, Ctensor)
        myo_passive = 2.0*diff(phi_m*W_myo, Ctensor)
        collagen_passive = 2.0*diff(phi_c*W_collagen, Ctensor)
        if self.parameters["incompressible"]:
            incomp_stress = \
                -self.parameters["pressure_variable"]*inv(Ctensor)
        else:
            # 2*d[Kappa/2*(J-1)^2]/dC = Kappa*(J-1)*J*C^-1.
            J = self.J()
            incomp_stress = \
                self.parameters["Kappa"]*(J - 1.0)*J*inv(Ctensor)
        return (Ctensor, bulk_passive, myo_passive,
                collagen_passive, incomp_stress)

    def stress(self,hsl):
        Ctensor, bulk_passive, myo_passive, collagen_passive, \
            incomp_stress = self._passive_stress_components()
        f0 = self.parameters["fiber"]
        fiber_strain = 0.5*(inner(f0, Ctensor*f0) - 1.0)
        # Preserve Sff as the myofiber passive-stress stimulus used downstream.
        Sff = inner(f0, myo_passive*f0)
        passive_total_stress = bulk_passive + myo_passive + \
            collagen_passive + incomp_stress
        # The six-value legacy API has no collagen slot.  Keep bulk_passive as
        # the matrix output by including both ground and collagen constituents.
        matrix_passive = bulk_passive + collagen_passive
        return (passive_total_stress, Sff, myo_passive, matrix_passive,
                incomp_stress, fiber_strain)

    def passivestress(self,hsl):
        Ctensor, bulk_passive, myo_passive, collagen_passive, \
            incomp_stress = self._passive_stress_components()
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]
        e1 = Constant((1.0, 0.0, 0.0))
        e2 = Constant((0.0, 1.0, 0.0))
        e3 = Constant((0.0, 0.0, 1.0))
        TransMatrix = as_tensor(f0[i]*e1[j], (i,j)) + \
            as_tensor(s0[i]*e2[j], (i,j)) + \
            as_tensor(n0[i]*e3[j], (i,j))
        material_stress = bulk_passive + myo_passive + collagen_passive
        PK2_local = TransMatrix.T*material_stress*TransMatrix
        return PK2_local, incomp_stress

    def return_radial_vec_ratio(self):

        mesh = self.parameters["mesh"]
        s0 = self.parameters["sheet"]
        print s0[0]

        X = SpatialCoordinate(mesh)
        ratio = s0_evaluated.y()/s0_evaluated.x()

        return ratio

    def Umat(self):

        Fmat = self.Fmat()
        #Fmat = self.Fe()
        F0 = Fmat
        for j in range(15):
            F0 = 0.5* (F0 + inv(F0).T)
        R = F0
        return inv(R)*Fmat

    def kroon_law(self,FunctionSpace,step_size,kappa,binary_mask):

        mesh = self.parameters["mesh"]
        C = self.Cmat()
        f0 = self.parameters["fiber"]
        f = C*f0/sqrt(inner(C*f0,C*f0))
	f_proj = project(f,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
	for i in range(len(binary_mask)):
            f_array = f_proj.vector().get_local()[i*3:(i+1)*3]
            if binary_mask[i] == 1:

                f_proj.vector()[i*3] = f0.vector().get_local()[i*3]
                f_proj.vector()[i*3+1] = f0.vector().get_local()[i*3+1]
                f_proj.vector()[i*3+2] = f0.vector().get_local()[i*3+2]
        f_adjusted = 1./kappa * (f_proj - f0) * step_size
        f_adjusted = project(f_adjusted,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
        f_adjusted = interpolate(f_adjusted,FunctionSpace)

        return f_adjusted

    def eigen(self,T,dgs,dgv):

        mesh = self.parameters["mesh"]

        dofmap = dgs.dofmap()
        dofs = dofmap.dofs()

        eigval1 = Function(dgs)
        eigval2 = Function(dgs)
        eigval3 = Function(dgs)

        eigvec1 = Function(dgv)
        eigvec2 = Function(dgv)
        eigvec3 = Function(dgv)


        E1 = eigval1.vector().array()
        E2 = eigval2.vector().array()
        E3 = eigval3.vector().array()

        V1 = eigvec1.vector().array().reshape([len(dofs),3])
        V2 = eigvec2.vector().array().reshape([len(dofs),3])
        V3 = eigvec3.vector().array().reshape([len(dofs),3])


        Emax = E1
        Emin = E3
        E3rd = E2

        Vmax = V1
        Vmin = V2
        V3rd = V3

        print "calculating eigenvalue"

        F = T.vector().array()
        if all(np.equal(F,np.zeros(len(F)))) == True:

            return 'zero array'

        else:

            mesh1 = T.function_space().mesh()

            #print len(F)
            #print np.shape(F), len(dofs)
            #print mesh1

            gdim = mesh.geometry().dim()
            #print gdim

            # Get coordinates as len(dofs) x gdim array
            dofs_x = dgs.tabulate_dof_coordinates().reshape((-1, gdim))

            RC = F.reshape([len(dofs),3,3])

            RC = np.where(RC< 1e-10,0.,RC)

            for idx, (dof,x, v) in enumerate(zip(dofs, dofs_x, RC)):
                #print idx, dof, x, v
                [eigL,eigR] = np.linalg.eig(v)


                ls1 = eigL[0]
                ls2 = eigL[1]
                ls3 = eigL[2]


                lv1 = eigR[:,0]
                lv2 = eigR[:,1]
                lv3 = eigR[:,2]

                lv1 = lv1/np.dot(lv1, lv1)
                lv2 = lv2/np.dot(lv2, lv2)
                lv3 = lv3/np.dot(lv3, lv3)


                lsmax = max(ls1, ls2, ls3)

                lsmin = min(ls1, ls2, ls3)
                if lsmax == ls1:
                    Vmax[idx] = lv1
                    if lsmin==ls2:
                        ls3rd = ls3
                        Vmin[idx] = lv2
                        V3rd[idx] = lv3
                    elif lsmin == ls3:
                        ls3rd = ls2
                        Vmin[idx] = lv3
                        V3rd[idx] = lv2

                elif lsmax == ls2:
                    Vmax[idx] = lv2
                    if lsmin == ls1:
                        ls3rd = ls3
                        Vmin[idx] = lv1
                        V3rd[idx] = lv3
                    elif lsmin == ls3:
                        ls3rd = ls1
                        Vmin[idx] = lv3
                        V3rd[idx] = lv1

                elif lsmax == ls3:
                    Vmax[idx] = lv3
                    if lsmin == ls1:
                        ls3rd = ls2
                        Vmin[idx] = lv1
                        V3[idx] = lv2
                    elif lsmin == ls2:
                        ls3rd = ls1
                        Vmin[idx] = lv2
                        V3rd[idx] = lv1

                if Vmax[idx,0] < 0:
                    Vmax [idx,:] *= -1
                else:
                    pass

                Emax[idx] = lsmax
                Emin[idx] = lsmin
                E3rd[idx] = ls3rd

            maxvec = Function(dgv)
            maxvec.vector().set_local(Vmax.flatten())
            maxvec.vector().apply("insert")

            minvec = Function(dgv)
            minvec.vector().set_local(Vmin.flatten())
            minvec.vector().apply("insert")

            vec3rd = Function(dgv)
            vec3rd.vector().set_local(V3rd.flatten())
            vec3rd.vector().apply("insert")

            emax = Function(dgs)
            emin = Function(dgs)
            e3rd = Function(dgs)

            emax.vector().set_local(Emax.flatten())
            emax.vector().apply("insert")

            emin.vector().set_local(Emin.flatten())
            emin.vector().apply("insert")

            e3rd.vector().set_local(E3rd.flatten())
            e3rd.vector().apply("insert")


            return maxvec


    def stress_kroon(self,stress_tensor,FS,VFS,TFS,step_size,kappa):

        mesh = self.parameters["mesh"]
        f0 = self.parameters["fiber"]
        #inv_F = inv(self.Fmat())
        eigen = self.eigen(stress_tensor,FS,VFS)

        if eigen == 'zero array':
            f = f0
        else:
            #print "eigen"
            #print eigen.vector().get_local().reshape(24,3)[0:4]
            f = eigen
            f /= sqrt(inner(f,f))



        f_adjusted = 1./kappa * (f - f0) * step_size
        f_adjusted = project(f_adjusted,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
        #print 'f_adj before interpolate: '
        #print f_adjusted.vector().get_local()[0:4]
        f_adjusted = interpolate(f_adjusted,VFS)
        #print 'f_adj after interpolate: '
        #print f_adjusted.vector().get_local()[0:4]

        return f_adjusted

    def new_stress_kroon(self,stress_tensor,FunctionSpace,step_size,kappa,binary_mask):

        mesh = self.parameters["mesh"]
        PK2 = stress_tensor
        f0 = self.parameters["fiber"]
        f = PK2*f0/sqrt(inner(PK2*f0,PK2*f0))

	f_proj = project(f,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
        """for i in range(no_of_int_points):
            f_array = f_proj.vector().get_local()[i*3:(i+1)*3]
            if np.all(np.isnan(f_array)):
		f_proj.vector()[i*3] = f0.vector().get_local()[i*3]
                f_proj.vector()[i*3+1] = f0.vector().get_local()[i*3+1]
                f_proj.vector()[i*3+2] = f0.vector().get_local()[i*3+2]"""
	"""for i in range(len(binary_mask)):
            f_array = f_proj.vector().get_local()[i*3:(i+1)*3]
            if binary_mask[i] == 1:
                f_proj.vector()[i*3] = f0.vector().get_local()[i*3]
                f_proj.vector()[i*3+1] = f0.vector().get_local()[i*3+1]
                f_proj.vector()[i*3+2] = f0.vector().get_local()[i*3+2]"""

        """for index in np.arange(len(binary_mask)):
            if binary_mask[index] == 1:
                f.vector()[index*3] = f0.vector().get_local()[index*3]
                f.vector()[index*3+1] = f0.vector().get_local()[index*3+1]
                f.vector()[index*3+2] = f0.vector().get_local()[index*3+2]"""

        f_adjusted = 1./kappa * (f_proj - f0) * step_size
        f_adjusted = project(f_adjusted,VectorFunctionSpace(mesh,"DG",1),form_compiler_parameters={"representation":"uflacs"})
        f_adjusted = interpolate(f_adjusted,FunctionSpace)

        return f_adjusted


    def rand_walk(self,width):


        f0 = self.parameters["fiber"]
        for i in np.arange(np.shape(f0.vector().array())[0]/3):
            i = int(i)
            f0.vector()[i*3] = np.random.normal(f0.vector().array()[i*3],width)
            f0.vector()[i*3+1] = np.random.normal(f0.vector().array()[i*3+1],width)
            f0.vector()[i*3+2] = np.random.normal(f0.vector().array()[i*3+2],width)

        return f0
    
    def F_print(self,F):

        mesh = self.parameters["mesh"]
        
        #F = self.Fe()
        fs = TensorFunctionSpace(mesh, "DG", 0)
        fs._quad_scheme = 'default'
        fs_proj = project(F,fs,
                form_compiler_parameters={"representation":"uflacs"})

        F_projected = Function(fs)
        F_values = fs_proj.vector().get_local()

        # Get the mesh from F
        mesh = fs_proj.function_space().mesh()
        cell_dofs = fs_proj.function_space().dofmap().cell_dofs
        num_cells = mesh.num_cells()

        # Define the dimension of the deformation gradient tensor
        dim = 3  # Assuming 3D problem with 3x3 tensor


        cell_id = 10    
        dofs = cell_dofs(cell_id)
        # Extract the tensor values for this cell
        tensor_values = np.array(F_values[dofs]).reshape((dim, dim))

        return tensor_values

    def cycle_strain(self):

        mesh = self.parameters["mesh"]
        
        F = self.Fe()