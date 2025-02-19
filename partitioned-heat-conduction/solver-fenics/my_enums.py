from enum import Enum


class ProblemType(Enum):
    """
    Enum defines problem type. Details see above.
    """
    DIRICHLET = "Dirichlet"  # Dirichlet problem
    NEUMANN = "Neumann"  # Neumann problem


class DomainPart(Enum):
    """
    Enum defines which part of the domain [x_left, x_right] x [y_bottom, y_top] we compute.
    """
    FULL = 0 # full domain in simple interface case
    LEFT = 1  # left part of domain in simple interface case
    RIGHT = 2  # right part of domain in simple interface case
    CIRCULAR = 3  # circular part of domain in complex interface case
    RECTANGLE = 4  # domain excluding circular part of complex interface case
