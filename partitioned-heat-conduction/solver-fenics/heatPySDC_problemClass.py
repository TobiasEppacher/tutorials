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
    
    def __init__(self, mesh, functionSpace, couplingBoundary, remainingBoundary, t0=0.0):
        # Allow for fixing the boundary conditions for the residual computation
        # Necessary if imex-1st-order-mass is used
        self.fix_bc_for_residual = True
        
        # set mesh
        self.mesh = mesh
        
        # define function space for future reference
        self.V = functionSpace
        
        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_2d, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'mesh', 'functionSpace', 't0', localVars=locals(), readOnly=True
        )
        
        
        # Stiffness term (Laplace)
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        a_K = -1.0 * df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx

        # Mass term
        a_M = u * v * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # Define function parameters and boundary condition
        alpha = 3
        self.alpha = 3
        beta = 1.2
        self.beta = 1.2
        
        x_sp, y_sp, t_sp = sp.symbols(['x[0]', 'x[1]', 't'])
        u_D_sp = 1 + x_sp * x_sp + alpha * y_sp * y_sp + beta * t_sp
        self.u_D = Expression(sp.ccode(u_D_sp), degree=2, alpha=self.alpha, beta=self.beta, t=t0)
        self.f_N = Expression(sp.ccode(u_D_sp.diff(x_sp)), degree=1, alpha=alpha, t=t0)
        
        # define the Dirichlet boundary
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc1 = df.DirichletBC(self.V, self.u_D, remainingBoundary)
        self.bc2 = df.DirichletBC(self.V, self.u_D, couplingBoundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant(0), boundary)

        # set forcing term as expression
        self.g = df.Expression('beta - 2 - 2 * alpha', degree=2, alpha=self.alpha, beta=self.beta, t=t0)
        
        
    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor \cdot A) \vec{u} = \vec{rhs}`.
        """
        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.u_D.t = t

        self.bc1.apply(T, b.values.vector())
        self.bc2.apply(T, b.values.vector())
        self.bc1.apply(b.values.vector())
        self.bc2.apply(b.values.vector())

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
    
    def get_f_N(self):
        return self.f_N
    
    def get_u_D(self):
        return self.u_D