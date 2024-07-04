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
    
    def __init__(self, mesh, functionSpace, couplingBC, remainingBC, forcingTermExpr, couplingExpr, preciceRef, solutionExpr=None):
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
        
        if couplingBC is not None:
            self.K = df.assemble(a_K)
        else:
            # TODO: Make this case for Neumann BCs work
            class Left(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and near(x[0], 1.0)
            
            neumann_boundary_domain = df.MeshFunction('size_t', self.mesh, mesh.topology().dim() - 1)
            neumann_boundary_domain.set_all(0)
            Left().mark(neumann_boundary_domain, 1)
            
            ds = df.Measure('ds', domain=self.mesh, subdomain_data=neumann_boundary_domain)
            a_K += v * couplingExpr * ds(1)
            
            # Assemble stiffness matrix with specification of the different subdomains of the integrals
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
        
        # Update coupling expression
        dt = self.t_end - self.t_start
        dt_factor = (t - self.t_start) / dt
        
        read_data = self.precice.read_data(dt_factor * dt)
        self.precice.update_coupling_expression(self.coupling_expression, read_data)
        
        
        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.u_D.t = t

        # Time of the boundary condition is set in the precice loop
        self.remainingBC.apply(T, b.values.vector())
        self.remainingBC.apply(b.values.vector())
        
        # Coupling BC is only needed here for Dirichlet participant, Neumann BC is handled different
        if self.couplingBC is not None:
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

        if self.couplingBC is not None:
            # Coupling BC needs to point to correct time
            dt = self.t_end - self.t_start
            dt_factor = (t - self.t_start) / dt
            
            read_data = self.precice.read_data(dt_factor * dt)
            self.precice.update_coupling_expression(self.coupling_expression, read_data)

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
    
    
'''
##### Test the class #####

from problem_setup import get_geometry
from my_enums import DomainPart

from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

import matplotlib.pyplot as plt

# Get the mesh and boundaries
domain_mesh, coupling_boundary, remaining_boundary = get_geometry(DomainPart.LEFT)

# Define function space using mesh
V = FunctionSpace(domain_mesh, 'P', 2)
V_g = VectorFunctionSpace(domain_mesh, 'P', 1)
W = V_g.sub(0).collapse()

alpha = 3
beta = 1.2
x_sp, y_sp, t_sp = sp.symbols(['x[0]', 'x[1]', 't'])
u_D_sp = 1 + x_sp * x_sp + alpha * y_sp * y_sp + beta * t_sp

u_D = Expression(sp.ccode(u_D_sp), degree=2, alpha=alpha, beta=beta, t=0)
f = Expression(sp.ccode(u_D_sp.diff(t_sp) - u_D_sp.diff(x_sp).diff(x_sp) - u_D_sp.diff(y_sp).diff(y_sp)), degree=2, alpha=alpha, beta=beta, t=0)

remaining_BC = DirichletBC(V, u_D, remaining_boundary)
coupling_BC = DirichletBC(V, u_D, coupling_boundary)

# Set timestep size
pySDC_dt = 0.1

# initialize level parameters
level_params = dict()
level_params['restol'] = 1e-12
level_params['dt'] = pySDC_dt

# initialize step parameters
step_params = dict()
step_params['maxiter'] = 10

# initialize sweeper parameters
sweeper_params = dict()
sweeper_params['quad_type'] = 'RADAU-RIGHT'
sweeper_params['num_nodes'] = 4

# initialize problem parameters
problem_params = dict()
problem_params['mesh'] = domain_mesh
problem_params['functionSpace'] = V
problem_params['t0'] = 0.0
problem_params['couplingBC'] = coupling_BC
problem_params['remainingBC'] = remaining_BC
problem_params['solutionExpr'] = u_D
problem_params['forcingTermExpr'] = f

# initialize controller parameters
controller_params = dict()
controller_params['logger_level'] = 30

# fill description dictionary for easy step instantiation
description = dict()
description['problem_class'] = fenics_heat_2d
description['problem_params'] = problem_params
description['sweeper_class'] = imex_1st_order_mass
description['sweeper_params'] = sweeper_params
description['level_params'] = level_params
description['step_params'] = step_params

# Controller for time stepping
controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

# Reference to problem class for easy access to exact solution
P = controller.MS[0].levels[0].prob

t_start = 0.1
t_end = 0.5

u_init = P.u_exact(t_start)
u_end, _ = controller.run(u_init, t0=t_start, Tend=t_end)
u_ref = P.u_exact(t_end)

# Compute the error
err = abs(u_end - u_ref) / abs(u_ref)

print(err)

# Plot the solution
#plot(u_end.values)
#plt.show()

'''

