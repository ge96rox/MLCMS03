import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def logistic_func(r, x):
    return r * x * (1 - x)


def lorenz_func(t, state, rho, sigma, beta):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def plot_logistic_map_bifurcation(r_range=[0, 4], x_range=[0, 1],
                                  resolution=100, iteration=100):
    x = np.linspace(x_range[0], x_range[1], resolution)
    r = np.linspace(r_range[0], r_range[1], resolution)
    # plt.figure()
    plt.xlim(r_range[0], r_range[1])
    for i in range(iteration):
        x = logistic_func(r, x)
        plt.plot(r, x)

    plt.xlabel("r")
    plt.ylabel("x")
    plt.title("Bifurcation diagram for the logistic map")
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
        ax.set_title(rf'$\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {round(beta, 2)}')
    plt.show()


