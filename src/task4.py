import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def logistic_func(r, x):
    return r * x * (1 - x)


def lorenz_func(t, state, rho, sigma, beta):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def plot_logistic_map_bifurcation(r_range=[0, 4], resolution=1000, iteration=1000, last_iter_plot=100):
    x = 1e-5 * np.ones(resolution)
    lyapunov = np.zeros(resolution)
    r = np.linspace(r_range[0], r_range[1], resolution)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    for i in range(iteration):
        x = logistic_func(r, x)
        lyapunov += np.log(abs(r - 2 * r * x) + 1e-20)
        if i >= iteration - last_iter_plot:
            ax1.plot(r, x, ',')
    ax1.set_xlim(r_range[0], r_range[1])
    ax1.set_xlabel("r")
    ax1.set_ylabel("x")
    ax1.set_title("Bifurcation diagram for logistic map")

    ax2.plot(r[lyapunov < 0], lyapunov[lyapunov < 0] / iteration, ',k', markersize=1,  alpha=.5)

    ax2.plot(r[lyapunov >= 0], lyapunov[lyapunov >= 0] / iteration, ',r', markersize=1, alpha=.5)

    ax2.grid(color='grey', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("r")
    ax2.set_ylabel(r"$\lambda$")
    ax2.set_title("Maximal Lyapunov exponent for logistic map")
    ax2.set_ylim(-5, 1)
    plt.tight_layout()
    plt.show()


def plot_lorenz_burfication(y0_list, rho, sigma, beta):
    t = np.linspace(0, 1000, 10000)
    t_span = [t[0], t[-1]]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    for y0 in y0_list:
        sol = solve_ivp(lorenz_func, t_span, y0, t_eval=t, args=(rho, sigma, beta))
        ax.plot(sol.y[0], sol.y[1], sol.y[2],
                label="x = {0}, y = {1},  z = {2} ".format(y0[0], y0[1], y0[2]))
        ax.legend()
        ax.set_title(rf'$\sigma$ = {sigma}, $\rho$ = {round(rho, 3)}, $\beta$ = {round(beta, 3)}')
    plt.show()