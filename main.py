from utils import *
from optimization_solver import OptimizationDiscreteSolver
from optimization_ivp_solver import OptimizationIVPDiscreteSolver
import matplotlib.pyplot as plt
import scipy


def plot(y, u, dt):
    import matplotlib.pyplot as plt

    plt.plot(np.arange(0, dt * len(y[0, :]), dt), y[0, :], label="y[0]")
    plt.plot(np.arange(0, dt * len(y[0, :]), dt), y[1, :], label="y[1]")
    plt.plot(np.arange(0, dt * len(u[0, :]), dt), u[0, :], label="u")
    plt.legend()
    plt.show()


def plot_analytical_vs_numerical(y_sol, u_sol, y, u, dt):
    # Your implementation of plot_analytical_vs_numerical goes here
    # Subplot 1
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(0, dt * len(y[0, :]), dt), y[0, :], label="y[0] numerical")
    plt.plot(
        np.arange(0, dt * len(y_sol[0, :]), dt), y_sol[0, :], label="y[0] analytical"
    )
    plt.legend()
    # Subplot 2
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(0, dt * len(y[0, :]), dt), y[1, :], label="y[1] numerical")
    plt.plot(
        np.arange(0, dt * len(y_sol[1, :]), dt), y_sol[1, :], label="y[1] analytical"
    )
    plt.legend()
    # Subplot 3
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(0, dt * len(u[0, :]), dt), u[0, :], label="u numerical")
    plt.plot(np.arange(0, dt * len(u_sol[0, :]), dt), u_sol[0, :], label="u analytical")
    plt.legend()
    plt.show()


def defineProblem():
    m = float(input("Enter m: "))
    g = float(input("Enter g: "))
    c = float(input("Enter c: "))
    alpha = float(input("Enter alpha: "))
    numSteps = int(input("Enter numSteps: "))
    T = float(input("Enter T: "))
    x0 = np.zeros((2, 1))

    def f(y, u):
        return np.array(
            [
                float(y[1]),
                (float(u) - c * float(y[1]) - m * g) / m,
            ]
        ).flatten()

    def j(y, u):
        return (T / numSteps) * (alpha / 2 * np.sum(u**2) - np.sum(y[0, :]))

    return m, g, c, alpha, numSteps, T, x0, f, j


def solution(m, g, c, alpha, T):
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

    return y_solution, u_solution


# Reward hacking!!! Better solutions wrt numerical objective are worse wrt analytical objective.


def main():
    m, g, c, alpha, numSteps, T, x0, f, j = defineProblem()
    solver = OptimizationIVPDiscreteSolver(
        x0=x0,
        T=T,
        numSteps=numSteps,
        f=f,
        j=j,
        mode="average",
        method="Nelder-Mead",
        solve_for_lambdas=False,
    )
    y, u = solver.solve()

    y_solution, u_solution = solution(m, g, c, alpha, T)
    y_sol, u_sol = discrete_solution(numSteps, solver, y_solution, u_solution)

    plot_analytical_vs_numerical(y_sol, u_sol, y, u, solver.dt)
    print(
        f"Optimal loss: {scipy.integrate.quad(lambda t: alpha*u_solution(t)**2-y_solution(t)[0], 0, T)[0]}"
    )
    print(f"My loss: {solver.j(y, u)}")


def discrete_solution(numSteps, solver, y_solution, u_solution):
    y_sol = np.zeros((2, numSteps))
    u_sol = np.zeros((1, numSteps))
    for i in range(numSteps):
        y_sol[:, i] = y_solution(i * solver.dt)
        u_sol[:, i] = u_solution(i * solver.dt)
    return y_sol, u_sol


if __name__ == "__main__":
    main()
