import numpy as np
from sympy import Min, Max, sqrt, Function, sin, cos
from sympy.utilities.lambdify import implemented_function


def sympy_dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def sympy_dot_3d(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sympy_ndot(a, b):
    return a[0] * b[0] - a[1] * b[1]


def sympy_norm(a):
    return sqrt(sympy_dot(a, a))


def sympy_norm_3d(a):
    return sqrt(sympy_dot_3d(a, a))


def sympy_clamp(x, bottom_value, top_value):
    return Min(Max(x, bottom_value), top_value)


def sign_func(x):
    y = np.ones_like(x)
    y[x == 0] = 0
    y[x < 0] = -1
    return y
f_sign = implemented_function(Function('f_sign', real=True), sign_func)


def heaviside_func(x):
    y = np.ones_like(x)
    y[x <= 0] = 0
    return y
f_heaviside = implemented_function(Function('f_heaviside', real=True), heaviside_func)


def min3_func(x):
    return np.minimum.reduce(x)
f_min3 = implemented_function(Function('f_min3', real=True), min3_func)


def max3_func(x):
    return np.maximum.reduce(x)
f_max3 = implemented_function(Function('f_max3', real=True), max3_func)


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


