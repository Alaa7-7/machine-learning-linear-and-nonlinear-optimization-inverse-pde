import numpy as np

# Simple 1D heat equation
# u_t = k * u_xx

k = 0.1

length = 1.0
time = 0.5

nx = 20
nt = 50

dx = length / nx
dt = time / nt

# Initial temperature
u = np.zeros(nx + 1)

# Hot temperature at the left side
u[0] = 100.0

# Time loop
for n in range(nt):

    old_u = u.copy()

    for i in range(1, nx):

        u[i] = old_u[i] + k * dt / (dx * dx) * (
            old_u[i + 1] - 2 * old_u[i] + old_u[i - 1]
        )

    # Boundary conditions
    u[0] = 100.0
    u[nx] = 0.0

print("Heat equation experiment")
print("-------------------------")
print("k =", k)
print("Temperature at the middle =", u[nx // 2])
print("Temperature at the right side =", u[nx])