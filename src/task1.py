import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig


def plot_phase_potrait(matrix, pts, range_x=(-1, 1), range_y=(-1, 1), num_grid_points=10):
    """function that plot phase_potrait of linear system X'=AX

    Parameters
    ----------
    matrix: np.ndarray
        matrix A in equation X'=AX
    pts: list of list
        initial points
    range_x : tuple
        range of x
    range_y: int
        range of y
    num_grid_points: int
        number of point within range x/y
    Returns
    -------
    None
    """
    eigenvalues = eig(matrix)[0]

    x, y = np.meshgrid(np.linspace(range_x[0], range_x[1], num_grid_points),
                       np.linspace(range_y[0], range_y[1], num_grid_points))
    u, v = np.zeros_like(x), np.zeros_like(y)

        
    u = matrix[0][0] * x + matrix[0][1] * y
    v = matrix[1][0] * x + matrix[1][1] * y

    plt.figure(figsize=(6,6))
    plt.streamplot(x, y, u, v, start_points=pts, density=35, linewidth=2, arrowsize=2)
    plt.quiver(x, y, u, v)
    plt.axis('square')
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(r'$\alpha={0}$, $\lambda_1={1}$, $\lambda_2={2}$'.format(matrix[0][0], eigenvalues[0], eigenvalues[1]))
    
    plt.show()
