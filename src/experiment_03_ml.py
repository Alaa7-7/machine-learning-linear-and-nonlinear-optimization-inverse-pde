import numpy as np

# Machine learning for inverse PDE

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


# Training data
k_train = np.linspace(0.05, 0.15, 21)

temperature_train = []

for k in k_train:

    solution = solve_heat(k)

    temperature_train.append(solution[nx // 2])

temperature_train = np.array(temperature_train)


# Polynomial features
X = np.column_stack([
    np.ones(len(temperature_train)),
    temperature_train,
    temperature_train ** 2,
    temperature_train ** 3
])

y = k_train


# Ridge regularization
alpha = 0.001

A = X.T @ X + alpha * np.eye(X.shape[1])
b = X.T @ y

coefficients = np.linalg.solve(A, b)


# Measured temperature
true_solution = solve_heat(true_k)
measured_temperature = true_solution[nx // 2]


# Predict k
x_new = np.array([
    1.0,
    measured_temperature,
    measured_temperature ** 2,
    measured_temperature ** 3
])

predicted_k = x_new @ coefficients


print("Machine learning experiment")
print("True k =", true_k)
print("Measured temperature =", measured_temperature)
print("ML predicted k =", predicted_k)
print("Prediction error =", abs(predicted_k - true_k))