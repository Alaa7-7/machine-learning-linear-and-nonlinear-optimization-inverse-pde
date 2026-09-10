import numpy as np
from scipy.optimize import minimize

# Stochastic optimization for inverse PDE

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


# Clean temperature
true_solution = solve_heat(true_k)

clean_temperature = true_solution[nx // 2]


# Create several possible measurements
# using random noise

np.random.seed(42)

noise_level = 0.05

number_of_measurements = 20

measurements = []

for i in range(number_of_measurements):

    noise = np.random.normal(
        0,
        noise_level * clean_temperature
    )

    measured_temperature = clean_temperature + noise

    measurements.append(measured_temperature)

measurements = np.array(measurements)


# Objective function
def objective(x):

    k = x[0]

    errors = []

    for i in range(number_of_measurements):

        solution = solve_heat(k)

        predicted_temperature = solution[nx // 2]

        error = (
            predicted_temperature
            - measurements[i]
        ) ** 2

        errors.append(error)

    # Average error
    average_error = np.mean(errors)

    return average_error


# Initial guess
initial_guess = [0.0923]


# Bounds
bounds = [(0.05, 0.15)]


# Stochastic optimization
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


print("Stochastic programming experiment")
print("True k =", true_k)
print("Clean temperature =", clean_temperature)
print("Noise level =", noise_level * 100, "%")
print("Number of measurements =", number_of_measurements)
print("Initial ML guess =", initial_guess[0])
print("Stochastic estimated k =", estimated_k)
print("Parameter error =", parameter_error)
print("Average objective error =", result.fun)