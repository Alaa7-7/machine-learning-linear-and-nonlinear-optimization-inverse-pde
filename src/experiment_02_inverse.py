import numpy as np

# Inverse heat equation
# We know a temperature measurement
# and try to estimate k

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


# Create a measured temperature
true_solution = solve_heat(true_k)
measured_temperature = true_solution[nx // 2]


# Try different values of k
k_values = np.linspace(0.05, 0.15, 101)

best_k = None
best_error = float("inf")

for k in k_values:

    solution = solve_heat(k)

    predicted_temperature = solution[nx // 2]

    error = abs(predicted_temperature - measured_temperature)

    if error < best_error:

        best_error = error
        best_k = k


print("Inverse heat equation experiment")
print("True k =", true_k)
print("Measured temperature =", measured_temperature)
print("Estimated k =", best_k)
print("Error =", best_error)