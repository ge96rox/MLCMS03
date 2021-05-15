import sys
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def plot_phase_potrait_Andronov_Hopf(alpha, range_x=(-2, 2), range_y=(-2, 2), num_grid_points=10):
    '''
    plot phase potrait for model.
    

    '''  

    x, y = np.meshgrid(np.linspace(range_x[0], range_x[1], num_grid_points),
                       np.linspace(range_y[0], range_y[1], num_grid_points))
    u, v = np.zeros_like(x), np.zeros_like(y)

    # Andronov_Hopf normal form   
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

    
def Andronov_Hopf_model(t,y):
    #alpha = 1  
    
    x1,x2 =y
    u = 1 * x1 - x2 - x1*(x1**2+x2**2)          
    v = x1 + 1 * x2 - x2*(x1**2+x2**2)
    return[u,v]
    

def plot_orbit_with_time(starting_point, t_0 = 0, t_end = 15, num_step = 150):
    
    t = np.linspace(t_0,t_end,num_step)
    y0 = np.array(starting_point)
    soln = solve_ivp(Andronov_Hopf_model, t_span=[t[0],t[-1]],y0=y0, t_eval=t)
    
    
    
    fig = plt.figure(figsize=(20,7))
    fig.suptitle('3D View of Orbit along Time Starting from' +str(starting_point))
    
    ax1 = fig.add_subplot(121,projection='3d')
    ax1.scatter3D(soln.y[0],soln.y[1],soln.t)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('time')
    
    
  
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.scatter3D(soln.y[0],soln.y[1],soln.t)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('time');
    ax2.view_init(90,0)
 
    
    
    
    
