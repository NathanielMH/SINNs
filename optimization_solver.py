from numpy import ndarray
from typing import Callable
import numpy as np
from scipy.optimize import minimize

from utils import deviate, modeToFunction, diff, H, getLambdas


class OptimizationDiscreteSolver:
    def __init__(
        self,
        x0: ndarray,
        T: float,
        numSteps: int,
        f: Callable,
        j: Callable,
        deviate: Callable = deviate,
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
        self.lossODE = self.getLossODE()
        self.deviate = deviate
        self.mode = mode
        self.method = method
        self.solve_for_lambdas = solve_for_lambdas
        # self.set_initial_value = set_initial_value

    def getLossODE(self) -> Callable:
        def lossODE(y, u) -> float:
            return sum(
                [
                    self.deviate(
                        (y[:, i + 1] - y[:, i]) / self.dt,
                        self.f(
                            modeToFunction[self.mode](y[:, i], y[:, i + 1]),
                            u[:, i],
                        ),
                    )
                    for i in range(self.numSteps - 1)
                ]
            )

        # Modify the loss term.

        return lossODE

    # yu = np.array([y_0,y_1, ..., y_(self.dimY-1), u_(0), u_(1), ..., u_(self.dimU-1)).flatten()
    def totalLoss(self) -> Callable:
        def loss(yu) -> float:
            y, u = (
                yu[: -self.numSteps * self.dimU].reshape(self.dimY, self.numSteps - 1),
                yu[-self.numSteps * self.dimU :].reshape(self.dimU, self.numSteps),
            )
            y = np.concatenate([self.x0, y], axis=1)
            return self.j(y, u) + self.lossODE(y, u)

        return loss

    def solve(self) -> ndarray:
        res = minimize(
            self.totalLoss(),
            np.random.rand(
                (self.dimU + self.dimY) * (self.numSteps - 1) + self.dimU
            ).flatten(),
            method=self.method,
        )
        y, u = (
            res.x[: -self.numSteps * self.dimU].reshape(self.dimY, self.numSteps - 1),
            res.x[-self.numSteps * self.dimU :].reshape(self.dimU, self.numSteps),
        )
        y = np.concatenate([self.x0, y], axis=1)
        self.y, self.u = y, u
        if self.solve_for_lambdas:
            self.lambdas = getLambdas(y, u)
            return y, u, self.lambdas
        return y, u

    # Gets set of (y_i,u_i) points for the system.
