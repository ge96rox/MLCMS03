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
    plt.plot(origin_x[0], origin_x[1], 'o', label='bifurcation starting point')
    plt.xlabel('alpha')
    plt.ylabel('x')
    plt.xlim(-1,5)
    plt.title(description_text)
    plt.legend()
    
    
def phase_portrait_plot(alpha, system_id = 2, range_x=(-2,2), range_y=(-1,1), num_grid_points=10):
    
    x, y = np.meshgrid(np.linspace(range_x[0], range_x[1], num_grid_points),
                       np.linspace(range_y[0], range_y[1], num_grid_points))
    u, v = np.zeros_like(x), np.zeros_like(y)
    
    if system_id == 1:
        u = alpha - x ** 2
        v = -y
    else:
        u = alpha - 2 * x ** 2 - 3
        v = -y
        
    plt.figure(figsize=(7, 7))
    plt.streamplot(x, y, u, v, linewidth=2, arrowsize=2)
    plt.quiver(x, y, u, v)
    plt.xlabel("x")
    plt.ylabel("y")
    if system_id == 1:
        plt.title(r'$\dot{x} = \alpha - x^2$'+', '+r'$\alpha={0}$'.format(alpha))
    else:
        plt.title(r'$\dot{x} = \alpha - 2x^2 - 3$'+', '+r'$\alpha={0}$'.format(alpha))

    plt.show()

