import numpy as np
from scipy.optimize import linprog

# Linear programming for inverse PDE

true_k = 0.1

# Temperature measured from the heat equation
measured_temperature = 11.487271887444848

# Simple linear relationship
# temperature = a * k + b

k1 = 0.05
k2 = 0.15

T1 = 0.0
T2 = 0.0


def solve_heat(k):

    length = 1.0
    time = 0.5

    nx = 20
    nt = 50

    dx = length / nx
    dt = time / nt

    u = np.zeros(nx + 1)

    u[0] = 100.0

    for n in range(nt):

        old_u = u.copy()

        for i in range(1, nx):

            u[i] = old_u[i] + k * dt / (dx * dx) * (
                old_u[i + 1]
                - 2 * old_u[i]
                + old_u[i - 1]
            )

        u[0] = 100.0
        u[nx] = 0.0

    return u


T1 = solve_heat(k1)[10]
T2 = solve_heat(k2)[10]


a = (T2 - T1) / (k2 - k1)
b = T1 - a * k1


# Linear prediction of temperature
# T = a * k + b

# We want to minimize the difference between
# predicted temperature and measured temperature.

# Variable:
# z = absolute error

c = np.array([0.0, 1.0])

A_ub = np.array([
    [a, -1.0],
    [-a, -1.0]
])

b_ub = np.array([
    measured_temperature - b,
    -measured_temperature + b
])

bounds = [
    (0.05, 0.15),
    (0, None)
]


result = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method="highs"
)


estimated_k = result.x[0]
error = abs(estimated_k - true_k)


print("Linear programming experiment")
print("True k =", true_k)
print("Measured temperature =", measured_temperature)
print("LP estimated k =", estimated_k)
print("Error =", error)