import numpy as np
from math import sin, cos
x0 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
eps = 0.000001
np.random.seed(42)


def function(x):
    res = 0.5 * x[0] ** 2
    for i in range(1, len(x)):
        res += 0.5 * ((cos(x[i]) + x[i - 1] - 1) ** 2)
    return res


def grad(x):
    res = np.zeros(10)
    res[0] += x[0]
    for i in range(1, len(x)):
        res[i] -= cos(x[i]) * sin(x[i])
        res[i - 1] += x[i - 1]
        res[i] -= x[i - 1] * sin(x[i])
        res[i - 1] += cos(x[i])
        res[i] += sin(x[i])
        res[i - 1] -= 1
    return res


def norm(x):
    res = 0
    for i in range(len(x)):
        res += x[i] ** 2
    res = res ** 0.5
    return res


def goldstein_search(x_k, p_k, max_iter = 100000, eps=0.000001):
    delta_1 = 0.35
    delta_2 = 0.65
    np.random.seed(42)
    tau = np.random.rand()
    teta_1 = 1.7
    teta_2 = 0.5
    t_l = 0
    t_r = 0
    i = 0
    f_k = function(x_k)
    g_k = grad(x_k)
    while i < max_iter:
        i += 1
        f_k1 = function(x_k + tau * p_k)
        cond_2 = f_k + delta_2 * np.dot(g_k.transpose(), p_k) * tau - f_k1 <= eps
        cond_1 = f_k1 - f_k + delta_1 * np.dot(g_k.transpose(), p_k) * tau <= eps
        if not cond_1: # cond_1 is violated
            t_r = tau
            tau = (1 - teta_2) * t_l + teta_2 * t_r
            continue
        if not cond_2: # cond_2 is violated
            t_l = tau
        if cond_1 and cond_2:
            return tau
        if t_r == 0:
            tau = teta_1 * tau
        else:
            tau = (1 - teta_2) * t_l + teta_2 * t_r
    return tau

def polak_ribiere(x0, eps=0.001):
    x = x0.copy()
    p = -grad(x0)
    k = 0
    while np.linalg.norm(grad(x)) > eps:
        a = goldstein_search(x, p)
        x_new = x + a * p
        b = np.dot(grad(x_new).transpose(), (grad(x_new) - grad(x))) / norm(grad(x)) ** 2
        p = -grad(x_new) + b * p
        k = k + 1
        x = x_new.copy()
    return (k, x)


numb_of_iter, point_of_optimum = polak_ribiere(x0)
print("Number of iterations:", numb_of_iter)
print("Point of optimum:", point_of_optimum)
print("Value of function at point of optimum:", function(point_of_optimum))





