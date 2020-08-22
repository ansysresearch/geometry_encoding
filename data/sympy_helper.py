import numpy as np
from sympy import Min, Max, sqrt, Function
from sympy.utilities.lambdify import implemented_function
import matplotlib.pyplot as plt

def sympy_dot(x, y):
    return x[0] * y[0] + x[1] * y[1]

def sympy_ndot(x, y):
    return x[0] * y[0] - x[1] * y[1]

def sympy_norm(x):
    return sqrt(sympy_dot(x, x))

def sympy_clamp(x, bottom_value, top_value):
    return Min(Max(x, bottom_value), top_value)

def sign_func(x):
    y = np.ones_like(x)
    y[x == 0] = 0
    y[x < 0] = -1
    return y
f_sign = implemented_function(Function('f_sign'), sign_func)

def heaviside_func(x):
    y = np.ones_like(x)
    y[x <= 0] = 0
    return y
f_heaviside = implemented_function(Function('f_heaviside'), heaviside_func)

def min3_func(x):
    return np.minimum.reduce(x)
f_min3 = implemented_function(Function('f_min3'), min3_func)

SIMPY_2_NUMPY_DICT = {
                       'Min': np.minimum,
                       'Max': np.maximum,
                       'Abs': np.abs,
                       'sin': np.sin,
                       'cos': np.cos,
                       'sqrt': np.sqrt,
                       #'sign': sign_func,
                       'Heaviside': heaviside_func
                      }

def plot_sdf(img, sdf, xticks=(-1, 1), yticks=(-1, 1), plot_eikonal=False, show=True):
    plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='binary')
    plt.gca().invert_yaxis()
    plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(xticks[0], xticks[1], 5))
    plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(yticks[0], yticks[1], 5))

    plt.subplot(2, 2, 2)
    plt.imshow(sdf, cmap='hot')
    #plt.colorbar()
    plt.contour(sdf, 30, colors='k')
    plt.gca().invert_yaxis()
    plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(xticks[0], xticks[1], 5))
    plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(yticks[0], yticks[1], 5))

    if plot_eikonal:
        dx, dy = (xticks[1] - xticks[0]) / img.shape[0],  (yticks[1] - yticks[0]) / img.shape[1]
        sdf_gradient = np.gradient(sdf)
        sdf_gradient_values = np.sqrt(sdf_gradient[0] ** 2 / dx ** 2+ sdf_gradient[1] ** 2 / dy ** 2)

        plt.subplot(2, 2, 3)
        plt.imshow(sdf_gradient_values)
        plt.gca().invert_yaxis()
        plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(xticks[0], xticks[1], 5))
        plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(yticks[0], yticks[1], 5))
        plt.colorbar()

    if show:
        plt.show()
