import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def logistic_func(r, x):
    return r * x * (1 - x)


def lorenz_func(t, state, rho, sigma, beta):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def plot_logistic_map_bifurcation(r_range=[0, 4], x_range=[0, 1],
                                  resolution=1000, iteration=1000,
                                  last_iter_plot=100, save=False):
    x = 1e-5 * np.ones(resolution)
    lyapunov = np.zeros(resolution)
    r = np.linspace(r_range[0], r_range[1], resolution)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    for i in range(iteration):
        x = logistic_func(r, x)
        lyapunov += np.log(abs(r - 2 * r * x) + 1e-20)
        if i >= iteration - last_iter_plot:
            ax1.plot(r, x, ',')
    ax1.set_xlim(r_range[0], r_range[1])
    ax1.set_ylim(x_range[0], x_range[1])
    ax1.set_xlabel("r")
    ax1.set_ylabel("x")
    ax1.set_title("Bifurcation diagram for logistic map")

    ax2.plot(r[lyapunov < 0], lyapunov[lyapunov < 0] / iteration, ',k', markersize=1, alpha=.5)

    ax2.plot(r[lyapunov >= 0], lyapunov[lyapunov >= 0] / iteration, ',r', markersize=1, alpha=.5)

    ax2.grid(color='grey', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("r")
    ax2.set_ylabel(r"$\lambda$")
    ax2.set_title("Maximal Lyapunov exponent for logistic map")
    ax2.set_ylim(-5, 1)
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig("./img/task4_logistic_map_range{0}_to_{1}.pdf".format(r_range[0], r_range[1]), bbox_inches='tight')


def plot_lorenz_burfication(y0_list, rho, sigma, beta, sim_time, resolution, save=False):
    t = np.linspace(0, sim_time, resolution)
    t_span = [t[0], t[-1]]
    sols = []
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    for y0 in y0_list:
        sol = solve_ivp(lorenz_func, t_span, y0, t_eval=t, args=(rho, sigma, beta))
        sols.append(sol)
        ax.plot(sol.y[0], sol.y[1], sol.y[2],
                label="x = {0}, y = {1},  z = {2} ".format(y0[0], y0[1], y0[2]))
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(rf'$\sigma$ = {sigma}, $\rho$ = {round(rho, 3)}, $\beta$ = {round(beta, 3)}')
    first_time_idx = cal_first_time(sols[0], sols[1])
    plt.show()
    if save:
        fig.savefig("./img/task4_lorenz_{0}.pdf".format(round(rho)), bbox_inches='tight')
    if first_time_idx != np.inf:
        print("since time point {0}, the difference between the points on the"
              " trajectory larger than 1".format(round(first_time_idx * sim_time / resolution)))
    else:
        print("there does not exist a time point, "
              "since then the distance between points larger than 1")


def cal_first_time(sol0, sol1):
    sol0_x, sol0_y, sol0_z = sol0.y[0], sol0.y[1], sol0.y[2]
    sol1_x, sol1_y, sol1_z = sol1.y[0], sol1.y[1], sol1.y[2]
    sol0_x = sol0_x[:, np.newaxis]
    sol0_y = sol0_y[:, np.newaxis]
    sol0_z = sol0_z[:, np.newaxis]
    sol1_x = sol1_x[:, np.newaxis]
    sol1_y = sol1_y[:, np.newaxis]
    sol1_z = sol1_z[:, np.newaxis]
    sol0 = np.hstack((sol0_x, sol0_y, sol0_z))
    sol1 = np.hstack((sol1_x, sol1_y, sol1_z))
    diff = np.linalg.norm((sol0 - sol1), axis=1)
    first_time_idx = np.where(diff > 1)[0]
    if first_time_idx.size != 0:
        return first_time_idx[0]
    else:
        return np.inf


def plot_logistic_map_simulation(r_values, x_init=0.1, n=50, save=False):
    fig = plt.figure(figsize=(9, 15))

    n_values = np.linspace(0, n - 1, n)
    i = 0

    for r in r_values:
        x = x_init * np.ones(n)
        for j in range(1, n):
            x[j] = logistic_func(r, x[j - 1])

        ax = plt.subplot(len(r_values), 1, i + 1)
        i += 1

        ax.plot(n_values, x)
        ax.set_title("Logistic Map with x = {0}, r = {1}".format(x_init, r))
        ax.set(xlabel='n', ylabel='x', xlim=(0, n - 2))
        ax.grid()

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    if save:
        fig.savefig("./img/task4_simulation.pdf", bbox_inches='tight')
