import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from numpy.linalg import eig


def plot_phase_potrait_Andronov_Hopf(alpha, range_x=(-2, 2), range_y=(-2, 2), num_grid_points=10):
    '''
    plot phase potrait for model.
    

    '''  

    x, y = np.meshgrid(np.linspace(range_x[0], range_x[1], num_grid_points),
                       np.linspace(range_y[0], range_y[1], num_grid_points))
    u, v = np.zeros_like(x), np.zeros_like(y)

         
    u = alpha * x - y - x*(x**2+y**2)          
    v = x + alpha * y - y*(x**2+y**2)
    
    plt.figure(figsize=(9,9))
    plt.streamplot(x, y, u, v,linewidth=2, arrowsize=2)
    plt.quiver(x, y, u, v)
    plt.plot(0,0, marker="x",markersize=20)
    
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(r'$\alpha={0}$'.format(alpha))
    
    plt.show()

