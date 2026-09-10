import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Integer programming for inverse PDE

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


# Possible values of k
k_values = np.arange(0.05, 0.151, 0.01)


# Calculate the temperature for each possible k
temperatures = []

for k in k_values:

    solution = solve_heat(k)

    temperature = solution[nx // 2]

    temperatures.append(temperature)

temperatures = np.array(temperatures)


# We use one binary variable for each possible k.
#
# x[j] = 1 means that k_values[j] is selected.
# x[j] = 0 means that it is not selected.


# Objective:
# minimize the temperature error

errors = abs(
    temperatures - measured_temperature
)


# The optimization minimizes:
#
# sum(errors[j] * x[j])


c = errors


# Exactly one value of k must be selected.
constraint = LinearConstraint(
    np.ones((1, len(k_values))),
    1,
    1
)


# Binary variables
bounds = Bounds(
    np.zeros(len(k_values)),
    np.ones(len(k_values))
)


integrality = np.ones(len(k_values))


# Integer programming
result = milp(
    c=c,
    integrality=integrality,
    bounds=bounds,
    constraints=constraint
)


# Find the selected k
selected_index = np.argmax(result.x)

estimated_k = k_values[selected_index]

parameter_error = abs(
    estimated_k - true_k
)


print("Integer programming experiment")
print("True k =", true_k)
print("Measured temperature =", measured_temperature)
print("IP estimated k =", estimated_k)
print("Parameter error =", parameter_error)
print("Selected value =", estimated_k)