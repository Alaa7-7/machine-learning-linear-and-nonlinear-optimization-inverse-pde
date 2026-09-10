import numpy as np
from scipy.optimize import minimize

# Noise experiment for inverse PDE

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

        u[0] = 100.0
        u[nx] = 0.0

    return u


# Clean temperature
true_solution = solve_heat(true_k)

clean_temperature = true_solution[nx // 2]


# Noise levels
noise_levels = [0.0, 0.01, 0.05, 0.10]


print("Noise experiment")
print("----------------")
print("True k =", true_k)
print("Clean temperature =", clean_temperature)
print()


for noise_level in noise_levels:

    # Fixed seed for reproducible results
    np.random.seed(42)

    noise = np.random.normal(
        0,
        noise_level * clean_temperature
    )

    measured_temperature = clean_temperature + noise


    def objective(x):

        k = x[0]

        solution = solve_heat(k)

        predicted_temperature = solution[nx // 2]

        error = (
            predicted_temperature
            - measured_temperature
        ) ** 2

        return error


    # ML initial estimate
    initial_guess = [0.0923]

    result = minimize(
        objective,
        initial_guess,
        bounds=[(0.05, 0.15)],
        method="Nelder-Mead"
    )


    estimated_k = result.x[0]

    parameter_error = abs(
        estimated_k - true_k
    )


    print("Noise level =", noise_level * 100, "%")
    print("Measured temperature =", measured_temperature)
    print("Estimated k =", estimated_k)
    print("Parameter error =", parameter_error)
    print()
