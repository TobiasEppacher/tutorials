from fenics import *
import dolfin as df

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh

import sympy as sp


class fenics_heat_2d(ptype):
    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh
    
    def getDofCount(self):
        return len(Function(self.V).vector()[:])
    
    def __init__(self, mesh, functionSpace, forcingTermExpr, couplingBC, remainingBC, couplingExpr, preciceRef, solutionExpr):
        # Allow for fixing the boundary conditions for the residual computation
        # Necessary if imex-1st-order-mass is used
        self.fix_bc_for_residual = True
        
        # Set precice reference and coupling expression reference to update coupling boundary 
        # at every step within pySDC
        self.precice = preciceRef
        self.coupling_expression = couplingExpr
        self.t_start = 0.0
        self.t_end = 1.0
        
        # set mesh
        self.mesh = mesh
        
        # define function space for future reference
        self.V = functionSpace
        
        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_2d, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'mesh', 'functionSpace', localVars=locals(), readOnly=True
        )
        
        # Define Trial and Test function
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        
        # Mass term
        a_M = u * v * df.dx
        self.M = df.assemble(a_M)
        
        # Stiffness term (Laplace)
        a_K = -1.0 * df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
        self.K = df.assemble(a_K)         
        
        self.u_D = solutionExpr
        self.g = forcingTermExpr
        
        # define the homogeneous Dirichlet boundary
        def boundary(x, on_boundary):
            return on_boundary
        
        self.remainingBC = remainingBC
        self.couplingBC = couplingBC
        self.bc_hom = df.DirichletBC(self.V, df.Constant(0), boundary)
        
        
    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor \cdot A) \vec{u} = \vec{rhs}`.
        """
        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.u_D.t = t

        # Time of the boundary condition is set in the precice loop
        self.remainingBC.apply(T, b.values.vector())
        self.remainingBC.apply(b.values.vector())
        
        # Coupling BC is only needed here for Dirichlet participant, Neumann BC is handled different
        if self.couplingBC is not None:         
            # Coupling BC needs to point to correct time
            dt = self.t_end - self.t_start
            dt_factor = (t - self.t_start) / dt
            
            read_data = self.precice.read_data(dt_factor * dt)
            self.precice.update_coupling_expression(self.coupling_expression, read_data)   
            
            self.couplingBC.apply(T, b.values.vector())
            self.couplingBC.apply(b.values.vector())

        df.solve(T, u.values.vector(), b.values.vector())

        return u

    def eval_f(self, u, t):
        """
            The right-hand side.
        """
        f = self.dtype_f(self.V)

        self.K.mult(u.values.vector(), f.impl.values.vector())
        
        self.g.t = t
        
        f.expl = self.dtype_u(df.interpolate(self.g, self.V))
        f.expl = self.apply_mass_matrix(f.expl)

        return f
    
    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual

        Parameters
        ----------
        res : dtype_u
              Residual
        """
        self.bc_hom.apply(res.values.vector())
        return None
    
    def apply_mass_matrix(self, u):
        r"""
            The product :math:`M \vec{u}`.
        """

        me = self.dtype_u(self.V)
        self.M.mult(u.values.vector(), me.values.vector())

        return me

    def u_exact(self, t):
        self.u_D.t = t
        
        me = self.dtype_u(interpolate(self.u_D, self.V), val=self.V)
        return me

    def set_t_start(self, t_start):
        self.t_start = t_start
        
    def set_t_end(self, t_end):
        self.t_end = t_end

