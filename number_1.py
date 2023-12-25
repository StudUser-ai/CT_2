import numpy as np
# x = [1, -2, 4]
# x = [0.139762, 0.077109, -2.943324]
x = [8.655485, -2.859871, 6.345261]


def check_primal_feasibility(x):
    x1, x2, x3 = x
    within_constraint = True
    eps = 0.0001

    h = 11 * x1 ** 2 + 6 * x1 * x2 + 10 * x1 * x3 + 39 * x1 + 3 * x2 ** 2 + 10 * x2 * x3 + 31 * x2 + 12 * x3 ** 2 - 5 * x3 - 120
    if abs(h) > eps:
        within_constraint = False

    g_1 = -43.25 * x1 - 137.25 * x2 + 90.75 * x3 - 594
    if abs(g_1) > eps:
        within_constraint = False

    g_2 = 106.75 * x1 + 5.75 * x2 - 86.25 * x3 - 380.25
    if abs(g_2) > eps:
        within_constraint = False

    return within_constraint


x_collection = [[1, -2, 4], [0.139762, 0.077109, -2.943324], [8.655485, -2.859871, 6.345261]]
for x in x_collection:
    print(check_primal_feasibility(x))


# Since none of the functions lies within the constraint, it will not 
# Unused template of the remaining program
"""
f_1 = -16 * x1 ** 3 - 16 * x1 * x3 ** 2

f_2 = 20 * x2 ** 3 - 12 * x2 * x3 ** 2

f_3 = -16 * x1 ** 2 * x3 - 12 * x2 ** 2 * x3 - 8 * x3 ** 2

h_1 = 22 * x1 + 6 * x2 + 10 * x3 + 39

h_2 = 6 * x1 + 6 * x2 + 10 * x3 + 31

h_3 = 10 * x1 + 10 * x2 + 24 * x3 - 5

# Complementarity check

g_1 = -43.25 * x1 - 137.25 * x2 + 90.75 * x3 - 594
if abs(g_1) > eps:
    mu_1 = 0
    mu_1_flag = 1

g_2 = 106.75 * x1 + 5.75 * x2 - 86.25 * x3 - 380.25
if abs(g_2) > eps:
    mu_2 = 0
    mu_2_flag = 1

if g_1 > eps or g_2 > eps:
    global_flag = False

# Stationarity check

# First check
# ch_1 = f_1 +
# ch_2 = f_2 +
# ch_3 = f_3 +

print("Global flag is", global_flag)

"""

