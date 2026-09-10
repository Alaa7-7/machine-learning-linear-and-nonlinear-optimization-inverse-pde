# Final comparison of optimization methods

import numpy as np


# True value of k
true_k = 0.1


# Optimization methods only
methods = [
    "Linear Programming",
    "Nonlinear Optimization",
    "Integer Programming",
    "Convex Optimization",
    "Stochastic Programming"
]


# Estimated values from previous experiments
estimated_k = np.array([
    0.05000019751462799,
    0.100015703125,
    0.100000,
    0.100000,
    0.099438828125
])


# Calculate parameter error
errors = abs(estimated_k - true_k)


# Find the best optimization method
best_index = np.argmin(errors)

best_method = methods[best_index]
best_k = estimated_k[best_index]
best_error = errors[best_index]


# Print comparison
print("Final Comparison Experiment")
print()

print("True k =", true_k)
print()

print("Method                     Estimated k        Error")
print("----------------------------------------------------------")


for i in range(len(methods)):

    print(
        "{:<26} {:.9f}      {:.9f}".format(
            methods[i],
            estimated_k[i],
            errors[i]
        )
    )


print()
print("Best Optimization Method")
print("Method =", best_method)
print("Estimated k =", best_k)
print("Error =", best_error)


# Compare Machine Learning and Nonlinear Optimization
ml_estimated_k = 0.09230076967320817

ml_error = abs(
    ml_estimated_k - true_k
)

nlp_error = errors[1]


improvement = ml_error / nlp_error


print()
print("Machine Learning vs Nonlinear Optimization")
print("ML estimated k =", ml_estimated_k)
print("ML error =", ml_error)
print("NLP estimated k =", estimated_k[1])
print("NLP error =", nlp_error)
print("NLP improvement factor =", improvement)