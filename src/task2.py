import numpy as np
import matplotlib.pyplot as plt


def first_equation(alpha):
    alpha_ge_zero = np.where(alpha < 0, np.nan, alpha)
    return (0, 0), np.sqrt(alpha_ge_zero), -np.sqrt(alpha_ge_zero)


def second_equation(alpha):
    alpha_ge_three = np.where(alpha < 3, np.nan, alpha)
    return (3, 0), np.sqrt((alpha_ge_three - 3) / 2), -np.sqrt((alpha_ge_three - 3) / 2)


def bifurcation_plot(func, alpha, description_text):
    origin_x, pos_x, neg_x = func(alpha)
    plt.plot(alpha, pos_x, 'b-', label='stable equilibrium')
    plt.plot(alpha, neg_x, 'r--', label='unstable equilibrium')
    plt.plot(origin_x[0], origin_x[1], 'o', label='steady state')
    plt.xlabel('alpha')
    plt.ylabel('x')
    plt.title(description_text)
    plt.legend()
