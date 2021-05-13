import numpy as np
import matplotlib.pyplot as plt


def phase_portrait_plot(
        matrix,
        alpha,
        x_min=-1,
        x_max=1,
        x_num=10,
        y_min=-1,
        y_max=1,
        y_num=10):
    x = np.linspace(x_min, x_max, x_num)
    y = np.linspace(y_min, y_max, y_num)
    xv, yv = np.meshgrid(x, y)
    u = np.empty([y_num, x_num], dtype=float)
    v = np.empty([y_num, x_num], dtype=float)
    for row in range(y_num):
        for col in range(x_num):
            xy = np.array([xv[row, col], yv[row, col]])
            uv = matrix @ xy
            u[row, col] = uv[0]
            v[row, col] = uv[1]
    plt.quiver(xv, yv, u, v)
    plt.streamplot(x, y, u, v, color='blue')
    plt.title('alpha =' + str(alpha))
    plt.show()
