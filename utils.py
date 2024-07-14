from typing import Callable
import numpy as np
from numpy import ndarray

modeToFunction = {
    "average": lambda x, y: (x + y) / 2,
    "min": lambda x, y: x,
    "max": lambda x, y: y,
}


def diff(f: Callable, h: float = 1e-7) -> Callable:
    """
    Returns the derivative of a function f using the central difference method.

    Args:
        - f: The function to differentiate.
        - h: The step size to use in the central difference method.

    Returns:
        - The derivative of f.
    """

    def df(x: float) -> float:
        return (f(x + h) - f(x - h)) / (2 * h)

    return df


def H(j: Callable, f: Callable) -> Callable:
    """
    Returns the Hamiltonian function of a system.

    Args:
        - j: The functional to minimize of the system.
        - f: The forcing function of the system (ODE).

    Returns:
        - The Hamiltonian function of the system.
    """

    def h(y, u, lmbd):
        return j(y, u) + lmbd.T @ f(y, u)

    return h


def deviate(a: ndarray, b: ndarray) -> float:
    return np.linalg.norm(a - b)


def getLambdas(y: ndarray, u: ndarray) -> ndarray:
    # Use lambda_i+1- lambda_i = -H_y(y_i, u_i, lambda_i) depending on the mode
    # could be -H_y(y_i+1, u_i+1, lambda_i+1)
    # or -H_y((y_i+1 + y_i)/2, (u_i+1 + u_i)/2, (lambda_i+1 + lambda_i)/2)
    ...
