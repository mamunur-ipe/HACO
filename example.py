import numpy as np
from HACO import HACO


"""
Example usage of the HACO optimizer with the Ackley Function.
Learn more about the Ackley Function: https://www.sfu.ca/~ssurjano/ackley.html
"""
# Define the dimension of the function
d = 8

def objective_function(xx):
    """
    Ackley function for optimization testing.

    It is a multi-modal function commonly used as a performance test problem
    in optimization. The global minimum is at the origin, where f(x) = 0.

    Args:
        xx: A list or numpy array representing the solution vector.

    Returns:
        The function value.
    """

    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = np.sum(np.square(xx))  # Sum of squares
    sum2 = np.sum(np.cos(c * xx))  # Sum of cosines
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# Define bounds and variable types
lb = [-5] * d  # Lower bounds of the variables
ub = [5] * d   # Upper bounds of the variables
var_types = ['float'] * d

# Instantiate the HACO optimizer
optimizer = HACO(
    objective_function=objective_function,
    lb=lb,
    ub=ub,
    var_types=var_types,
    population=30,
    probability_random_chicks=0.1,
    max_step=0.99,
    min_step=0.01,
    max_iteration=1000,
    plot_results=True,  # Enable result plotting
    verbose=False       # Disable iteration details
)

# Run the optimizer
y_best, x_best = optimizer.run()

# Print the results
print("\nOptimization Results:")
print(f"Best objective value: {y_best:.6f}")
print(f"Best solution: {x_best}")
