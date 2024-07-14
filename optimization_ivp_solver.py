from numpy import ndarray
from typing import Callable
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from utils import deviate, modeToFunction, diff, H, getLambdas


class OptimizationIVPDiscreteSolver:
    def __init__(
        self,
        x0: ndarray,
        T: float,
        numSteps: int,
        f: Callable,
        j: Callable,
        mode: str = "average",
        method: str = "Nelder-Mead",
        solve_for_lambdas: bool = False,
        dimU: int = 1,
    ) -> None:
        self.dt = T / numSteps
        self.dimU = dimU
        self.x0 = x0
        self.dimY = x0.shape[0]
        self.T = T
        self.numSteps = numSteps
        self.f = f
        self.j = j
        self.mode = mode
        self.method = method
        self.solve_for_lambdas = solve_for_lambdas

    # yu = np.array([y_0,y_1, ..., y_(self.dimY-1), u_(0), u_(1), ..., u_(self.dimU-1)).flatten()
    def totalLoss(self) -> Callable:
        def loss(u) -> float:
            y = solve_ivp(
                lambda t, y, u=u: self.f(y, u[int(t / (self.T / self.numSteps))]),
                (0, self.T),
                self.x0.flatten(),
                t_eval=np.linspace(0, self.T, self.numSteps),
            ).y
            return self.j(y, u)

        return loss

    def solve(self) -> ndarray:
        res = minimize(
            self.totalLoss(),
            np.random.rand((self.dimU) * (self.numSteps + 1)).flatten(),
            method=self.method,
        )
        u = res.x
        y = solve_ivp(
            lambda t, y, u=u: self.f(y, u[int(t / (self.T / self.numSteps))]),
            (0, self.T),
            self.x0.flatten(),
            t_eval=np.linspace(0, self.T, self.numSteps),
        ).y
        u = u.reshape(self.dimU, self.numSteps + 1)[:, :-1]

        self.y, self.u = y, u
        if self.solve_for_lambdas:
            self.lambdas = getLambdas(y, u)
            return y, u, self.lambdas
        return y, u

    # Gets set of (y_i,u_i) points for the system.
