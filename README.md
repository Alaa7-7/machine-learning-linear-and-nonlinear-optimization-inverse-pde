### Machine Learning and Linear and Nonlinear Optimization for an Inverse PDE Problem

## 1. Project Overview

This project studies a simple inverse PDE problem using the 1D heat equation.

The main unknown parameter is "k".

The project uses Machine Learning and different optimization methods to estimate the value of "k" from a temperature measurement.

The project is written in simple Python and uses simple mathematical ideas.

----------------------------------------------------------------------------------------------------------------

## 2. Main Question

The main question of this project is:

Can Machine Learning help optimization methods estimate an unknown parameter in an inverse PDE problem?

The project also studies how the result changes when the measurements contain noise.

---------------------------------------------------------------------------------------------------------

## 3. Heat Equation

```
The project starts with a simple 1D heat equation:

u_t = k * u_xx

Here:

- "u" = temperature
- "t" = time
- "x" = position
- "k" = heat parameter

The value of "k" is the parameter that we want to estimate in the inverse problem.
```

---------------------------------------------------------------------------------------------------------------

## 4. Numerical Method

```
The second derivative with respect to position is approximated using a simple finite difference:

u_xx = (u[i+1] - 2*u[i] + u[i-1]) / dx^2

The heat equation then becomes:

u[i] = old_u[i] + k * dt / dx^2 *
       (old_u[i+1] - 2*old_u[i] + old_u[i-1])

Here, "i" is the position index.

For example:

i = 1 , first internal position
i = 2 , second internal position

The values "i-1", "i", and "i+1" represent neighboring positions.
```
------------------------------------------------------------------------------------------------

## 5. Inverse PDE Problem

```
In the forward problem, "k" is known and the temperature is calculated.

In the inverse problem, the temperature is known and "k" must be estimated.

The true value used in the experiments is:

k = 0.1

The temperature at the middle of the domain is:

T_measured = 11.487271887444848

The goal is to find an estimated value of "k" that gives a temperature close to this measurement.

The parameter error is calculated as:

Error = |k_estimated - k_true|

A smaller error means a better estimate.
```

--------------------------------------------------------------------------------------------------------------


## 6. Project Experiments

```
The project contains 10 experiments.

Experiment 01 - Heat Equation

The first experiment solves the 1D heat equation.

The value of "k" is known: k = 0.1

The temperature at the middle is: 11.487271887444848

Purpose

This experiment creates the basic PDE model used in the rest of the project.
```
--------------------------------------------------------------------------------------------------------------------------

## Experiment 02 - Inverse PDE

```
The second experiment tries to estimate "k".

Different values of "k" are tested.

For each value, the predicted temperature is calculated.

The error is:

Error = |T_predicted - T_measured|

The value with the smallest error is selected.

Result

True k = 0.1
Estimated k = 0.1
Error = 0.0

Purpose

This experiment shows how an unknown PDE parameter can be estimated from a temperature measurement.
```
----------------------------------------------------------------------------------------------------------------

## Experiment 03 - Machine Learning

```
The third experiment uses a simple polynomial regression model with Ridge regularization.

The model learns the relationship between temperature and "k".

The model is:

k = a0 + a1*T + a2*T^2 + a3*T^3

The Ridge parameter is:

alpha = 0.001

Result

True k = 0.1
ML predicted k = 0.0923007697
Prediction error = 0.0076992303

Purpose

Machine Learning gives an initial estimate of "k".

This estimate is later used as an initial guess for optimization.
```
--------------------------------------------------------------------------------------------------------

## Experiment 04 - Linear Programming

```
The fourth experiment uses Linear Programming.

A simple linear relationship is used:

T = a * k + b

The optimization tries to minimize the temperature error.

Result

True k = 0.1
LP estimated k = 0.0500001975
Error = 0.0499998025

The result is not very accurate.

Purpose

This experiment shows that a simple linear approximation may not be enough for this inverse PDE problem.

This motivates the use of Nonlinear Optimization.
```

---------------------------------------------------------------------------------------------------


## Experiment 05 - Nonlinear Optimization

```
The fifth experiment uses Nonlinear Optimization.

The objective function is:

f(k) = (T(k) - T_measured)^2

The goal is:

minimize f(k)

The initial guess comes from Machine Learning:

Initial guess = 0.0923

The bounds are:

0.05 <= k <= 0.15

Result

True k = 0.1
NLP estimated k = 0.1000157031
Prediction error = 0.0000157031

Purpose

This experiment shows how optimization can improve the Machine Learning estimate.
```

-------------------------------------------------------------------------------------------------------------


## Experiment 06 - Noise

```
The sixth experiment studies the effect of measurement noise.

The noise levels are:

0%
1%
5%
10%

The noisy measurement is created using:

T_measured = T_clean + noise

Results

Noise       Estimated k        Parameter Error

0%          0.100015703        0.000015703
1%          0.100304141        0.000304141
5%          0.101602109        0.001602109
10%         0.103116406        0.003116406

Observation

When the noise increases, the parameter error also increases.

Purpose

This experiment tests how the method behaves when the measurement is not perfect.
```

----------------------------------------------------------------------------------------------------------


## Experiment 07 - Integer Programming

```
The seventh experiment uses Integer Programming.

Possible values of "k" are:

0.05, 0.06, 0.07, ..., 0.15

A binary variable is used for each possible value.

x_i = 1 , the value is selected

x_i = 0 , the value is not selected

Exactly one value must be selected:

sum(x_i) = 1

The optimization chooses the value with the smallest temperature error.

Result

True k = 0.1
IP estimated k = 0.1
Parameter error = 0.0

Important Note

The exact result is possible because "0.1" is included in the list of possible values.

Therefore, this result does not mean that Integer Programming is always better than other methods.
```

--------------------------------------------------------------------------------------------


## Experiment 08 - Convex Optimization

```
The eighth experiment studies a convex-type squared-error objective.

The objective function is:

f(k) = (T(k) - T_measured)^2

The goal is:

minimize f(k)

The initial guess is:

0.08

The bounds are:

0.05 <= k <= 0.15

Result

Estimated k = 0.1
Parameter error = approximately 0

Important Note

The Python function "minimize()" with the Nelder-Mead method was used.

Nelder-Mead is a numerical optimization method.

It is not a solver designed specifically for Convex Programming.

Therefore, this experiment demonstrates the idea of a convex-type squared-error objective and its numerical minimization.
```

----------------------------------------------------------------------------------------------------------------------


## Experiment 09 - Stochastic Programming

```
The ninth experiment uses several noisy measurements.

The number of measurements is:

N = 20

The noise level is:

5%

Each noisy measurement is:

T_i = T_clean + noise_i

The error for each measurement is:

error_i = (T(k) - T_i)^2

The average error is:

f(k) = (1/N) * sum(error_i)

For 20 measurements:

f(k) = (1/20) * sum from i=1 to 20 of
       (T(k) - T_i)^2

Result

True k = 0.1
Stochastic estimated k = 0.099438828
Parameter error = 0.000561172

Purpose

This experiment shows how several noisy measurements can be used together.
```

-----------------------------------------------------------------------------------------------------------


## 7. Experiment 10 - Final Comparison

```
The final experiment compares the optimization methods.

The parameter error is:

Error = |k_estimated - k_true|
```

# Final Results

- | Method                   | Estimated k         | Error
  |--------------------------|---------------------|---------------
  | Linear Programming       | 0.050000198         | 0.049999802
  | Nonlinear Optimization   | 0.100015703         | 0.000015703
  | Integer Programming      | 0.100000000         | 0.000000000
  | Convex Optimization      | 0.100000000         | approximately 0
  | Stochastic Programming   | 0.099438828         | 0.000561172

-----------------------------------------------------------------------------------------------------------------------

## 8. Machine Learning vs Nonlinear Optimization

```
One of the main comparisons in the project is between Machine Learning and Nonlinear Optimization.

Machine Learning gives:

k_ML = 0.0923007697

with error:

Error_ML = 0.0076992303

Nonlinear Optimization gives:

k_NLP = 0.1000157031

with error:

Error_NLP = 0.0000157031

The improvement factor is:

Improvement = Error_ML / Error_NLP

The result is:

Improvement = 490.299

Therefore, in this experiment, the NLP error is about 490 times smaller than the ML error.

This is one of the main results of the project.
```

----------------------------------------------------------------------------------------------------

## 9. Main Project Idea

The project does not use Machine Learning to replace optimization.

Instead, Machine Learning gives an initial estimate.

```
The process is:

Inverse PDE
     |
     v
Machine Learning
     |
     v
Initial estimate of k
     |
     v
Nonlinear Optimization
     |
     v
Improved estimate of k

In this experiment:

ML estimate  = 0.0923

NLP estimate = 0.1000157

True value   = 0.1

The optimization improves the initial Machine Learning estimate.
```

-----------------------------------------------------------------------------------------------------


## 10. Meaning of i

```
The symbol "i" is used as an index.

Its meaning depends on the part of the code.

In the heat equation:

i = position index

For example:

i = 1 , first internal position
i = 2 , second internal position

In the stochastic experiment:

i = measurement index

For example:

i = 1 , first noisy measurement
i = 2 , second noisy measurement

In the final comparison:

i = method index

For example:

i = 0 , Linear Programming
i = 1 , Nonlinear Optimization
i = 2 , Integer Programming
i = 3 , Convex Optimization
i = 4 , Stochastic Programming

The symbol "i" is simply an index used to organize values.
```

-----------------------------------------------------------------------------------------------------------


## 11. Project Structure

```
project-6-ml-optimization-pde/
|
+-- src/
|   +-- experiment_01_heat.py
|   +-- experiment_02_inverse.py
|   +-- experiment_03_ml.py
|   +-- experiment_04_lp.py
|   +-- experiment_05_nlp.py
|   +-- experiment_06_noise.py
|   +-- experiment_07_integer.py
|   +-- experiment_08_convex.py
|   +-- experiment_09_stochastic.py
|   +-- experiment_10_comparison.py
|
+-- data/
|
+-- results/
|   +-- experiment_01_explanation.txt
|   +-- experiment_02_explanation.txt
|   +-- experiment_03_explanation.txt
|   +-- experiment_04_explanation.txt
|   +-- experiment_05_explanation.txt
|   +-- experiment_06_explanation.txt
|   +-- experiment_07_explanation.txt
|   +-- experiment_08_explanation.txt
|   +-- experiment_09_explanation.txt
|   +-- experiment_10_explanation.txt
|   +-- final_comparison.txt
|
+-- README.md
```

---------------------------------------------------------------------------------------------------------


## 12. Python Libraries

```
The project mainly uses:

NumPy
SciPy

NumPy is used for arrays and numerical calculations.

SciPy is used for optimization methods.

The Machine Learning experiment uses a simple NumPy implementation of polynomial regression with Ridge regularization.

This keeps the project simple and avoids unnecessary libraries.
```

------------------------------------------------------------------------------------------------------------


## 13. How to Run the Experiments

```
Open Git Bash in the project folder.

Run an experiment with:

python src/experiment_01_heat.py

For example:

python src/experiment_05_nlp.py

The final comparison can be run with:

python src/experiment_10_comparison.py
```

---------------------------------------------------------------------------------------------------------

## 14. Final Conclusion

This project studies a simple inverse PDE problem using the 1D heat equation.

The unknown parameter is "k".

Different Machine Learning and optimization methods are tested.

The experiments show that different methods can produce different estimates.

Machine Learning provides a useful initial estimate.

Linear Programming gives a less accurate result in the simple linear experiment.

Nonlinear Optimization gives a very accurate continuous estimate.

Integer Programming can find the exact value when the true value is included in the candidate list.

The convex-type experiment demonstrates the use of a squared-error objective.

Stochastic Programming shows how several noisy measurements can be used together.

```
The most important result is the comparison between Machine Learning and Nonlinear Optimization.

Machine Learning gives:

k = 0.0923007697

Nonlinear Optimization gives:

k = 0.1000157031

The NLP error is about 490 times smaller than the ML error in this experiment.

Overall, the project demonstrates a simple combination of:

PDE
Inverse Problem
Machine Learning
Linear Optimization
Nonlinear Optimization
Integer Programming
Convex-type Optimization
Stochastic Programming

The main idea is simple:

Machine Learning can provide an initial estimate, and optimization can improve the estimate.
```

## 15 What I Learned

```
In this project, I learned how to use a simple PDE model to study an inverse problem.

I learned how to:

- solve a simple heat equation using numerical methods,
- estimate an unknown parameter from a measured temperature,
- use Machine Learning to get an initial estimate,
- use optimization methods to improve the estimate,
- compare different optimization methods,
- study the effect of measurement noise.

The main idea I learned is that Machine Learning can give a useful initial estimate, and optimization can improve this estimate.

This project also helped me understand how mathematics, numerical methods, Machine Learning, and optimization can work together in one problem.
```
