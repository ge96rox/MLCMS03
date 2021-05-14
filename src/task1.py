import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

def plot_phase_potrait(A, pts, range_x = (-1,1), range_y = (-1,1), num_grid_points = 10):
    '''
    plot phase potrait for model.
    

    '''
    eigenvalues = eig(A)[0]

    X, Y = np.meshgrid(np.linspace(range_x[0], range_x[1], num_grid_points), 
                       np.linspace(range_y[0], range_y[1], num_grid_points))
    U, V = np.zeros_like(X), np.zeros_like(Y)
   

    grid = np.meshgrid(X, Y)

    for i in range(num_grid_points):
        for j in range(num_grid_points):
            
            x, y = X[i, j], Y[i, j]
                        
            U[i,j] = A[0][0]*x+A[0][1]*y
            V[i,j] = A[1][0]*x+A[1][1]*y
            


    plt.streamplot(X, Y, U, V,start_points = pts,density=35,linewidth=2, arrowstyle='->', arrowsize = 2)
    plt.axis('square')
    plt.axis([-1, 1, -1, 1])
    plt.title(r'$\alpha={0}$, $\lambda_1={1}$, $\lambda_2={2}$'.format(A[0][0],eigenvalues[0],eigenvalues[1]))
    plt.show()

