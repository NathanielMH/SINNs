import pytest
import numpy as np
from optimization_solver import OptimizationDiscreteSolver
from optimization_ivp_solver import OptimizationIVPDiscreteSolver

# test example from PDF.


def plot(y, u, dt):
    import matplotlib.pyplot as plt

    plt.plot(np.arange(0, dt * len(y[0, :]), dt), y[0, :], label="y[0]")
    plt.plot(np.arange(0, dt * len(y[0, :]), dt), y[1, :], label="y[1]")
    plt.plot(np.arange(0, dt * len(u[0, :]), dt), u[0, :], label="u")
    plt.legend()
    plt.show()


def test_solver():
    m = 75
    g = 9.81
    c = 0.1
    numSteps = 100
    T = 10
    x0 = np.array([0.0, 0.0]).reshape(-1, 1)
    alpha = 1

    def j(y, u):
        return (T / numSteps) * (alpha / 2 * np.sum(u**2) - np.sum(y[0, :]))

    def f(y, u):
        return np.array(
            [
                float(y[1]),
                (float(u) - c * float(y[1]) - m * g) / m,
            ]
        ).flatten()

    beta = 2000
    solver = OptimizationDiscreteSolver(
        x0=x0,
        T=T,
        numSteps=numSteps,
        f=f,
        j=j,
        mode="average",
        method="L-BFGS-B",
        solve_for_lambdas=False,
        deviate=lambda x, y: beta * np.linalg.norm(x - y) ** 2,
    )
    y, u = solver.solve()

    def u_solution(t):
        return m / (alpha * c**2) * np.exp(c * (t - T) / m) - 1 / (alpha * c) * (
            t - T + m / c
        )

    def y_solution(t):
        a = m / (2 * alpha * c**3) * np.exp(-T * c / m)
        b = m / c**2 * (m * g - T / (alpha * c) - 2 * a * c)
        return np.array(
            [
                1 / c * (T / (alpha * c) - m * g) * (m / c * np.exp(-c * t / m) + t)
                + a * m / c * (np.exp(-c * t / m) + np.exp(c * t / m))
                - t**2 / (2 * alpha * c**2)
                + b,
                1 / c * (T / (alpha * c) - m * g) * (1 - np.exp(-c * t / m))
                - a * (np.exp(-c * t / m) + np.exp(c * t / m))
                - t / (alpha * c**2),
            ]
        ).flatten()

    y_sol = np.zeros((2, numSteps))
    u_sol = np.zeros((1, numSteps))
    for i in range(numSteps):
        y_sol[:, i] = y_solution(i * solver.dt)
        u_sol[:, i] = u_solution(i * solver.dt)
    print(f"Optimal loss: {solver.j(y_sol, u_sol)}")
    print(f"My loss: {solver.j(y, u)}")

    plot(y_sol, u_sol, solver.dt)
    plot(y, u, solver.dt)
    print(solver.lossODE(y, u))
    print(solver.j(y, u))
    assert (y[0, 0], y[1, 0]) == pytest.approx((0.0, 0.0))
    assert u.shape == (1, numSteps)
    assert y.shape == (2, numSteps)


def test_ivp_solver():
    m = 75
    g = 9.81
    c = 0.1
    numSteps = 100
    T = 10
    x0 = np.array([0.0, 0.0]).reshape(-1, 1)
    alpha = 1

    def f(y, u):
        return np.array(
            [
                float(y[1]),
                (float(u) - c * float(y[1]) - m * g) / m,
            ]
        ).flatten()

    def j(y, u):
        return (T / numSteps) * (alpha / 2 * np.sum(u**2) - np.sum(y[0, :]))

    beta = 2000
    solver = OptimizationIVPDiscreteSolver(
        x0=x0,
        T=T,
        numSteps=numSteps,
        f=f,
        j=j,
        mode="average",
        method="L-BFGS-B",
        solve_for_lambdas=False,
    )
    y, u = solver.solve()

    def u_solution(t):
        return m / (alpha * c**2) * np.exp(c * (t - T) / m) - 1 / (alpha * c) * (
            t - T + m / c
        )

    def y_solution(t):
        a = m / (2 * alpha * c**3) * np.exp(-T * c / m)
        b = m / c**2 * (m * g - T / (alpha * c) - 2 * a * c)
        return np.array(
            [
                1 / c * (T / (alpha * c) - m * g) * (m / c * np.exp(-c * t / m) + t)
                + a * m / c * (np.exp(-c * t / m) + np.exp(c * t / m))
                - t**2 / (2 * alpha * c**2)
                + b,
                1 / c * (T / (alpha * c) - m * g) * (1 - np.exp(-c * t / m))
                - a * (np.exp(-c * t / m) - np.exp(c * t / m))
                - t / (alpha * c**2),
            ]
        ).flatten()

    y_sol = np.zeros((2, numSteps))
    u_sol = np.zeros((1, numSteps))
    for i in range(numSteps):
        y_sol[:, i] = y_solution(i * solver.dt)
        u_sol[:, i] = u_solution(i * solver.dt)
    print(f"Optimal loss: {solver.j(y_sol, u_sol)}")
    print(f"My loss: {solver.j(y, u)}")

    plot(y_sol, u_sol, solver.dt)
    plot(y, u, solver.dt)
    print(solver.j(y, u))
    assert (y[0, 0], y[1, 0]) == pytest.approx((0.0, 0.0))
    assert u.shape == (1, numSteps)
    assert y.shape == (2, numSteps)


def main():
    test_solver()
    test_ivp_solver()


if __name__ == "__main__":
    main()
