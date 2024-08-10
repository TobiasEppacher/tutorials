"""
The basic example is taken from "Langtangen, Hans Petter, and Anders Logg. Solving PDEs in Python: The FEniCS
Tutorial I. Springer International Publishing, 2016."

The example code has been extended with preCICE API calls and mixed boundary conditions to allow for a Dirichlet-Neumann
coupling of two separate heat equations.

The original source code can be found on https://github.com/hplgit/fenics-tutorial/blob/master/pub/python/vol1/ft03_heat.py.

Heat equation with Dirichlet conditions. (Dirichlet problem)
  u'= Laplace(u) + f  in the unit square [0,1] x [0,1]
  u = u_C             on the coupling boundary at x = 1
  u = u_D             on the remaining boundary
  u = u_0             at t = 0
  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha

Heat equation with mixed boundary conditions. (Neumann problem)
  u'= Laplace(u) + f  in the shifted unit square [1,2] x [0,1]
  du/dn = f_N         on the coupling boundary at x = 1
  u = u_D             on the remaining boundary
  u = u_0             at t = 0
  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha

For information on the partitioned heat conduction problem using higher-order implicit Runge-Kutta methods see
* "Wullenweber, Nikola. High-order time stepping schemes for solving partial differential equations with FEniCS. Bachelor's thesis at Technical University of Munich, 2021. URL: https://mediatum.ub.tum.de/1621360"
* "Vinnitchenko, Niklas. Evaluation of higher-order coupling schemes with FEniCS-preCICE. Bachelor's thesis at Technical University of Munich, 2023. URL: https://mediatum.ub.tum.de/1732367"

The implementation of the higher-order implicit Runge-Kutta methods is based on: https://github.com/NikoWul/FenicsIrksome
"""

from __future__ import print_function, division
from fenics import Function, FunctionSpace, Expression, Constant, DirichletBC, TrialFunction, TestFunction, \
    File, solve, grad, inner, dx, interpolate, VectorFunctionSpace, MeshFunction, MPI
from fenicsprecice import Adapter
from errorcomputation import compute_errors
from my_enums import ProblemType, DomainPart
import argparse
import numpy as np
from problem_setup import get_geometry
import sympy as sp

from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from heat_pySDC_problemClass import fenics_heat_2d

def determine_gradient(V_g, u, flux):
    """
    compute flux following http://hplgit.github.io/INF5620/doc/pub/fenics_tutorial1.1/tu2.html#tut-poisson-gradu
    :param V_g: Vector function space
    :param u: solution where gradient is to be determined
    :param flux: returns calculated flux into this value
    """

    w = TrialFunction(V_g)
    v = TestFunction(V_g)

    a = inner(w, v) * dx
    L = inner(grad(u), v) * dx
    solve(a == L, flux)



parser = argparse.ArgumentParser(description="Solving heat equation for simple or complex interface case")
parser.add_argument("participantName", help="Name of the solver.", type=str, choices=[p.value for p in ProblemType])
parser.add_argument("-e", "--error-tol", help="set error tolerance", type=float, default=10**-8,)

args = parser.parse_args()
participant_name = args.participantName

# Time step size
# Can be anything smaller then the preCICE time window size
pySDC_dt = 0.25/1

# preCICE error tolerance
# Error is bounded by coupling accuracy. In theory we would obtain the analytical solution.
error_tol = args.error_tol

alpha = 3  # parameter alpha
beta = 1.2  # parameter beta
temporal_deg = 3  # temporal degree of the manufactured solution

if participant_name == ProblemType.DIRICHLET.value:
    problem = ProblemType.DIRICHLET
    domain_part = DomainPart.LEFT
elif participant_name == ProblemType.NEUMANN.value:
    problem = ProblemType.NEUMANN
    domain_part = DomainPart.RIGHT
    exit("Neumann problem not supported with pySDC")

domain_mesh, coupling_boundary, remaining_boundary = get_geometry(domain_part)

# Define function space using mesh
V = FunctionSpace(domain_mesh, 'P', 2)
V_g = VectorFunctionSpace(domain_mesh, 'P', 1)
W = V_g.sub(0).collapse()


##################################################################
# Problem definition and preCICE initialization
##################################################################

# Define boundary conditions
# create sympy expression of manufactured solution
x_sp, y_sp, t_sp = sp.symbols(['x[0]', 'x[1]', 't'])
u_D_sp = 1 + x_sp * x_sp + alpha * y_sp * y_sp + beta * (t_sp ** temporal_deg)
u_D = Expression(sp.ccode(u_D_sp), degree=2, alpha=alpha, beta=beta, t=0)
u_D_function = interpolate(u_D, V)
f_sp = u_D_sp.diff(t_sp) - u_D_sp.diff(x_sp).diff(x_sp) - u_D_sp.diff(y_sp).diff(y_sp)
forcing_expr = Expression(sp.ccode(f_sp), degree=2, alpha=alpha, beta=beta, t=0)


if problem is ProblemType.DIRICHLET:
    # Define flux in x direction
    f_N = Expression(sp.ccode(u_D_sp.diff(x_sp)), degree=1, alpha=alpha, t=0)
    f_N_function = interpolate(f_N, W)

# Define initial value
u_n = interpolate(u_D, V)
u_n.rename("Temperature", "")


precice, precice_dt, initial_data = None, 0.0, None

# Initialize the adapter according to the specific participant
precice = Adapter(adapter_config_filename="precice-adapter-config.json")

if problem is ProblemType.DIRICHLET:
    precice.initialize(coupling_boundary, read_function_space=V, write_object=f_N_function)
elif problem is ProblemType.NEUMANN:
    precice.initialize(coupling_boundary, read_function_space=W, write_object=u_D_function)

precice_dt = precice.get_max_time_step_size()
dt = Constant(0)
dt.assign(np.min([pySDC_dt, precice_dt]))

coupling_expression = precice.create_coupling_expression()


# Time-stepping
u_np1 = Function(V)
u_np1.rename("Temperature", "")
t = 0

# reference solution at t=0
u_ref = interpolate(u_D, V)
u_ref.rename("reference", " ")

# mark mesh w.r.t ranks
mesh_rank = MeshFunction("size_t", domain_mesh, domain_mesh.topology().dim())
if problem is ProblemType.NEUMANN:
    mesh_rank.set_all(MPI.rank(MPI.comm_world) + 4)
else:
    mesh_rank.set_all(MPI.rank(MPI.comm_world) + 0)
mesh_rank.rename("myRank", "")

# Generating output files
temperature_out = File("output/%s.pvd" % precice.get_participant_name())
ref_out = File("output/ref%s.pvd" % precice.get_participant_name())
error_out = File("output/error%s.pvd" % precice.get_participant_name())
ranks = File("output/ranks%s.pvd" % precice.get_participant_name())

# output solution and reference solution at t=0, n=0
n = 0
print("output u^%d and u_ref^%d" % (n, n))
ranks << mesh_rank

error_total, error_pointwise = compute_errors(u_n, u_ref, V)

# create buffer for output. We need this buffer, because we only want to
# write the converged output at the end of the window, but we also want to
# write the samples that are resulting from substeps inside the window
u_write = []
ref_write = []
error_write = []
# copy data to buffer and rename
uu = u_n.copy()
uu.rename("u", "")
u_write.append((uu, t))
uu_ref = u_ref.copy()
uu_ref.rename("u_ref", "")
ref_write.append(uu_ref)
err = error_pointwise.copy()
err.rename("err", "")
error_write.append(err)


if problem is ProblemType.DIRICHLET:
    flux = Function(V_g)
    flux.rename("Heat-Flux", "")


##################################################################
# Setup pySDC controller
##################################################################
'''
pySDC can be installed via `pip install pySDC`.
If you want to use the developer version, the source code repository can be cloned from "https://github.com/Parallel-in-Time/pySDC".
For more information on pySDC, see also "https://parallel-in-time.org/pySDC/"
'''

# initialize level parameters
level_params = dict()
level_params['restol'] = 1e-10
level_params['dt'] = pySDC_dt

# initialize step parameters
step_params = dict()
step_params['maxiter'] = 5

# initialize sweeper parameters
sweeper_params = dict()
sweeper_params['quad_type'] = 'LOBATTO'
sweeper_params['num_nodes'] = 4

# initialize problem parameters
problem_params = dict()
problem_params['mesh'] = domain_mesh
problem_params['function_space'] = V
problem_params['coupling_boundary'] = coupling_boundary
problem_params['remaining_boundary'] = remaining_boundary
problem_params['solution_expr'] = u_D
problem_params['forcing_term_expr'] = forcing_expr
problem_params['precice_ref'] = precice
problem_params['coupling_expr'] = coupling_expression
problem_params['participant_name'] = participant_name

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

##################################################################

    

#################################################################
# Coupling loop
#################################################################

while precice.is_coupling_ongoing():

    # write checkpoint
    if precice.requires_writing_checkpoint():
        precice.store_checkpoint(u_n, t, n)

        # output solution and reference solution at t_n+1 and substeps (read from buffer)
        print('output u^%d and u_ref^%d' % (n, n))
        for sample in u_write:
            temperature_out << sample

        for sample in ref_write:
            ref_out << sample

        for sample in error_write:
            error_out << error_pointwise

    precice_dt = precice.get_max_time_step_size()
    dt.assign(np.min([pySDC_dt, precice_dt]))

    # Update start and end time of the current preCICE time step within
    # the problem class.
    # This is necessary, that the coupling expression update, done in the problem class
    # itself is executed correctly.
    P.set_t_start(t)

    # Retrieve the result at the end of the timestep from pySDC
    # Possible statistics generated by pySDC can be accessed via the 'stats' variable if needed
    uend, stats = controller.run(u_n, t0=t, Tend=t + float(dt))
    
    # Update the buffer with the new solution values
    u_np1 = uend.values

    # Write data to preCICE according to which problem is being solved
    if problem is ProblemType.DIRICHLET:
        # Dirichlet problem reads temperature and writes flux on boundary to Neumann problem
        determine_gradient(V_g, u_np1, flux)
        flux_x = interpolate(flux.sub(0), W)
        precice.write_data(flux_x)
    elif problem is ProblemType.NEUMANN:
        # Neumann problem reads flux and writes temperature on boundary to Dirichlet problem
        precice.write_data(u_np1)

    precice.advance(dt)
    precice_dt = precice.get_max_time_step_size()

    # roll back to checkpoint
    if precice.requires_reading_checkpoint():
        u_cp, t_cp, n_cp = precice.retrieve_checkpoint()
        u_n.assign(u_cp)
        t = t_cp
        n = n_cp
        # empty buffer if window has not converged
        u_write = []
        ref_write = []
        error_write = []
    else:  # update solution
        u_n.assign(u_np1)
        t += float(dt)
        n += 1
        # copy data to buffer and rename
        uu = u_n.copy()
        uu.rename("u", "")
        u_write.append((uu, t))
        uu_ref = u_ref.copy()
        uu_ref.rename("u_ref", "")
        ref_write.append(uu_ref)
        err = error_pointwise.copy()
        err.rename("err", "")
        error_write.append(err)

    if precice.is_time_window_complete():
        u_ref = interpolate(u_D, V)
        u_ref.rename("reference", " ")
        error, error_pointwise = compute_errors(u_n, u_ref, V, total_error_tol=error_tol)
        print("n = %d, t = %.2f: L2 error on domain = %.3g" % (n, t, error))
    
    
# output solution and reference solution at t_n+1 and substeps (read from buffer)
print("output u^%d and u_ref^%d" % (n, n))
for sample in u_write:
    temperature_out << sample

for sample in ref_write:
    ref_out << sample

for sample in error_write:
    error_out << error_pointwise

# Hold plot
precice.finalize()