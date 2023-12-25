import numpy as np
from math import sin, cos
x0 = [1.1, 0.1, 1.1, 0.1, 1.1, 0.1, 1.1, 0.1, 1.1, 0.1]
eps = 0.000001
np.random.seed(42)


def function(x):
    res = 0
    for i in range(1, 6):
        res += (x[2 * i - 1 - 1] ** 2 + 10 * (x[2 * i - 1 - 1] ** 2 + x[2 * i - 1] ** 2 - 1) ** 2)
    return res


def grad(x):
    res = np.zeros(10)
    for i in range(5):
        res[2 * i] += 2 * x[2 * i]

        res[2 * i] += 40 * x[2 * i] ** 3
        res[2 * i + 1] += 40 * x[2 * i + 1] ** 3

        res[2 * i] += 40 * x[2 * i] * x[2 * i + 1] ** 2
        res[2 * i + 1] += 40 * x[2 * i + 1] * x[2 * i] ** 2

        res[2 * i] -= 40 * x[2 * i]
        res[2 * i + 1] -= 40 * x[2 * i + 1]
    return res


def norm(x):
    res = 0
    for i in range(len(x)):
        res += x[i] ** 2
    res = res ** 0.5
    return res


def wolfe_search(x_k, p_k, max_iter = 1000, eps=0.000001):
    delta_1 = 0.35
    delta_3 = 0.65
    np.random.seed(69)
    tau = np.random.rand()
    teta_1 = 1.15
    teta_2 = 0.5
    t_l = 0
    t_r = 0
    i = 0
    f_k = function(x_k)
    g_k = grad(x_k)
    while i < max_iter:
        i += 1
        f_k1 = function(x_k + tau * p_k)
        g_k1 = grad(x_k + tau * p_k)
        # cond_2 = delta_3 * np.dot(g_k.transpose(), p_k) - np.dot(g_k1.transpose(), p_k) <= eps
        # cond_1 = f_k1 - f_k - delta_1 * np.dot(g_k.transpose(), p_k) * tau <= eps

        cond_2 = delta_3 * np.dot(g_k, p_k) - np.dot(g_k1, p_k) <= eps
        cond_1 = f_k1 - f_k - delta_1 * np.dot(g_k.transpose(), p_k) * tau <= eps
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

def davidon_fletcher_powell(x0, eps=0.001):
    x = x0.copy()
    H = np.identity(10)
    k = 0
    g = grad(x)
    while norm(grad(x)) > eps and k < 50:
        d = np.dot(-H, g)
        tau_k = wolfe_search(x, d)
        x_new = x + tau_k * d
        g_new = grad(x_new)
        delta_k = x_new - x
        p_k = g_new - g
        delta_H = np.dot(delta_k, delta_k.transpose()) / np.dot(p_k.transpose(), delta_k)
        delta_H -= np.dot(np.dot(H, np.dot(p_k, p_k.transpose())), H) / np.dot(np.dot(p_k.transpose(), H), p_k)
        H = H + delta_H
        k = k + 1
        g = g_new.copy()
        x = x_new.copy()
    return (k, x)


numb_of_iter, point_of_optimum = davidon_fletcher_powell(x0)
print("Number of iterations:", numb_of_iter)
print("Point of optimum:", point_of_optimum)
print("Value of function at point of optimum:", function(point_of_optimum))

# Peculiar answer, maybe an unlucky initialization parameters
