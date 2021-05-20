import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve

from scipy.integrate import solve_ivp


def mu(b, I, mu0, mu1):
    """
    Recovery rate.
    
    """
    # recovery rate, depends on mu0, mu1, b
    mu = mu0 + (mu1 - mu0) * (b / (I + b))
    return mu


def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.
    """
    return beta / (d + nu + mu1)


def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.
    """
    c0 = b ** 2 * d * A
    c1 = b * ((mu0 - mu1 + 2 * d) * A + (beta - nu) * b * d)
    c2 = (mu1 - mu0) * b * nu + 2 * b * d * (beta - nu) + d * A
    c3 = d * (beta - nu)
    res = c0 + c1 * I + c2 * I ** 2 + c3 * I ** 3
    return res

def h_roots( mu0, mu1, beta, A, d, nu, b):
    
    c0 = b ** 2 * d * A
    c1 = b * ((mu0 - mu1 + 2 * d) * A + (beta - nu) * b * d)
    c2 = (mu1 - mu0) * b * nu + 2 * b * d * (beta - nu) + d * A
    c3 = d * (beta - nu)
    
    h = lambda I : c0 + c1 * I + c2 * I ** 2 + c3 * I ** 3
    x = np.array([0.0, 0.1])
    return fsolve(h,x)

def f(I, mu0, mu1, beta, A, d, nu, b):
    """
    Endemic equilibrium function for I.
    """
    r0 = R0(beta, d, nu, mu1)
    bigA= (d +nu + mu0)*(beta - nu)
    bigB= (d +nu +mu0 - beta)*A+(beta - nu)*(d +nu + mu1)*b
    bigC= (d +nu + mu1)*A*b*(1 - r0)
    
    s0=d+nu+mu0
    s1=d+nu+mu1
    
   
    
    #delta_0 = (beta - nu)**2*(s1**2)*(b**2) - 2*A*(beta - nu)*(beta*(mu1 - mu0)+ s0*(s1 - beta))*b +A**2*(beta - mu0)**2.
    #I = -bigB/(2*bigA)
    #I1 = (-bigB-np.sqrt(delta_0))/(2*A)
    #I2 = (-bigB+np.sqrt(delta_0))/(2*A)
    f = bigA * (I ** 2) + bigB * I + bigC
    
    return f
        

def f_roots( mu0, mu1, beta, A, d, nu, b):
    
    r0 = R0(beta, d, nu, mu1)
    bigA= (d +nu + mu0)*(beta - nu)
    bigB= (d +nu +mu0 - beta)*A+(beta - nu)*(d +nu + mu1)*b
    bigC= (d +nu + mu1)*A*b*(1 - r0)
    
    f = lambda I : bigA * I ** 2 + bigB * I + bigC
    x = np.array([0.0, 1])
    return fsolve(f,x)[1]


def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.
    
    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """
    S, I, R = y[:]
    m = mu(b, I, mu0, mu1)

    dSdt = A - d * S - (beta * S * I) / (S + I + R)
    dIdt = -(d + nu) * I - m * I + (beta * S * I) / (S + I + R)
    dRdt = m * I - d * R

    return [dSdt, dIdt, dRdt]


def sir_plots(
        random_state,
        t_0,
        t_end,
        rtol,
        atol,

        beta,
        A,
        d,
        nu,
        b,
        mu0,
        mu1,
        starting_point):
    # information
    print("Reproduction number R0=", R0(beta, d, nu, mu1))
    print('Globally asymptotically stable if beta <=d+nu+mu0. This is', beta <= d + nu + mu0)

    # simulation
    SIM0 = np.array(starting_point)

    NT = t_end - t_0
    time = np.linspace(t_0, t_end, NT)
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time,
                    args=(mu0, mu1, beta, A, d, nu, b), method='LSODA', rtol=rtol, atol=atol)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(sol.t, sol.y[0] - 0 * sol.y[0][0], label='1E0*susceptible')
    ax[0].plot(sol.t, 1e3 * sol.y[1] - 0 * sol.y[1][0], label='1E3*infective')
    ax[0].plot(sol.t, 1e1 * sol.y[2] - 0 * sol.y[2][0], label='1E1*removed')
    ax[0].set_xlim([0, 500])
    ax[0].legend();
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$S,I,R$")

    ax[1].plot(sol.t, mu(b, sol.y[1], mu0, mu1), label='recovery rate')
    ax[1].plot(sol.t, 1e2 * sol.y[1], label='1E2*infective')
    ax[1].set_xlim([0, 500])
    ax[1].legend();
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\mu,I$")

    I_h = np.linspace(-0., 0.05, 100)
    ax[2].plot(I_h, h(I_h, mu0, mu1, beta, A, d, nu, b));
    ax[2].plot(I_h, 0 * I_h, 'r:')
    # ax[2].set_ylim([-0.1,0.05])
    ax[2].set_title("Indicator function h(I)")
    ax[2].set_xlabel("I")
    ax[2].set_ylabel("h(I)")
    
    fr = f_roots(mu0, mu1, beta, A, d, nu, b)
    ax[2].plot(fr, 0 , marker='x', markersize =10 , linestyle='-', color='r')
        
    fig.tight_layout()


def plot_sir_trajectory(

        random_state,
        t_0,
        t_end,
        rtol,
        atol,

        beta,
        A,
        d,
        nu,
        b,
        mu0,
        mu1,

        starting_point):
    NT = t_end - t_0
    time = np.linspace(t_0, 3000, NT)

    cmap = ["BuPu", "Purples", "bwr"][1]

    SIM0 = np.array(starting_point)
    # b = 0.01
    fig = plt.figure(figsize=(20, 70))

    for i in range(21):
        sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time,
                        args=(mu0, mu1, beta, A, d, nu, b), method='DOP853', rtol=rtol, atol=atol)

        # draw the 3d plot
        ax = fig.add_subplot(11, 4, 2 * i + 1, projection="3d")
        ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=2, c=time)  ## CMAP not used here!!!
        ax.set_xlabel('S')
        ax.set_ylabel('I')
        ax.set_zlabel('R')
        ax.set_title("SIR trajectory with b= {0}".format(np.round(b, 3)))

        ax2 = fig.add_subplot(11, 4, 2 * i + 2)
        ax2.scatter(sol.y[0], sol.y[1], s=2, c=time)  ## CMAP not used here!!!
        ax2.set_xlabel('S')
        ax2.set_ylabel('I')
        
        ax2.plot(starting_point[0], starting_point[1], marker='X', linestyle='-', color='r')
        
        aspect = np.diff(ax2.get_xlim()) / np.diff(ax2.get_ylim())
        ax2.set_aspect(aspect)
        ax2.set_title("SI plane with b= {0}".format(np.round(b, 3)))

        b += 0.001

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    

