import numpy as np
from scipy.optimize import minimize

# Convex optimization for inverse PDE

true_k = 0.1

length = 1.0
time = 0.5

nx = 20
nt = 50

dx = length / nx
dt = time / nt


def solve_heat(k):

    u = np.zeros(nx + 1)

    # Initial temperature
    u[0] = 100.0

    for n in range(nt):

        old_u = u.copy()

        for i in range(1, nx):

            u[i] = old_u[i] + k * dt / (dx * dx) * (
                old_u[i + 1]
                - 2 * old_u[i]
                + old_u[i - 1]
            )

        # Boundary conditions
        u[0] = 100.0
        u[nx] = 0.0

    return u


# Measured temperature
true_solution = solve_heat(true_k)

measured_temperature = true_solution[nx // 2]


# Objective function
def objective(x):

    k = x[0]

    solution = solve_heat(k)

    predicted_temperature = solution[nx // 2]

    error = (
        predicted_temperature
        - measured_temperature
    ) ** 2

    return error


# Initial guess
initial_guess = [0.08]


# Bounds for k
bounds = [(0.05, 0.15)]


# Convex optimization
result = minimize(
    objective,
    initial_guess,
    bounds=bounds,
    method="Nelder-Mead"
)


estimated_k = result.x[0]

parameter_error = abs(
    estimated_k - true_k
)


print("Convex optimization experiment")
print("True k =", true_k)
print("Measured temperature =", measured_temperature)
print("Initial guess =", initial_guess[0])
print("Convex estimated k =", estimated_k)
print("Parameter error =", parameter_error)
print("Objective error =", result.fun)