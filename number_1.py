import numpy as np
# x = [1, -2, 4]
# x = [0.139762, 0.077109, -2.943324]
x = [8.655485, -2.859871, 6.345261]
x1, x2, x3 = x
print(x1, x2, x3)
global_flag = true
eps = 0.0001
h = 11 * x1 ** 2 + 6 * x1 * x2 + 10 * x1 * x3 + 39 * x1 + 3 * x2 ** 2 + 10 * x2 * x3 + 31 * x2 + 12 * x3 ** 2 -
5 * x3 - 120
print(h)
lamb = 0
lamb_flag = abs(11 * x1 ** 2 + 6 * x1 * x2 + 10 * x1 * x3 + 39 * x1 + 3 * x2 ** 2 + 10 * x2 * x3 + 31 * x2 +
12 * x3 ** 2 - 5 * x3 - 120) < eps
print(lamb_flag)
f_1 = -16 * x1 ** 3 - 16 * x1 * x3 ** 2
f_2 = 20 * x2 ** 3 - 12 * x2 * x3 ** 2
f_3 = -16 * x1 ** 2 * x3 - 12 * x2 ** 2 * x3 - 8 * x3 ** 2
h_1 = 22 * x1 + 6 * x2 + 10 * x3 + 39
h_2 = 6 * x1 + 6 * x2 + 10 * x3 + 31
