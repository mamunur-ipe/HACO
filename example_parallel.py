import numpy as np
from HACO import HACO
import statistics as stat
from joblib import Parallel, delayed
import multiprocessing as mp

# Set the number of CPU cores to use for parallel processing
num_cores = mp.cpu_count()

# Define the objective function (Ackley function)
def objective_function(xx):
    """
    Calculates the Ackley function value for a given solution vector.

    Args:
        xx: A list or numpy array representing the solution vector.

    Returns:
        The function value.
    """

    a = 20
    b = 0.2
    c = 2 * np.pi

    sum1 = 0
    sum2 = 0
    for ii in range(len(xx)):
        xi = xx[ii]
        sum1 += xi * xi
        sum2 += np.cos(c * xi)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / len(xx)))
    term2 = -np.exp(sum2 / len(xx))
    y = term1 + term2 + a + np.exp(1)
    return y

# Define problem parameters
d = 10  # Dimension of the problem
lb = [-5] * d  # Lower bounds
ub = [5] * d  # Upper bounds
var_types = ['float'] * d  # Variable types

# Set HACO algorithm parameters
population_size = 10 * d
probability_random_chicks = 0.01
max_step_size = 0.9
min_step_size = 0.01
max_iterations = 20 * d - 50
max_runtime = 3000

# Set the number of replications for statistical analysis
num_replications = 50

def run_haco():
    """
    Runs the HACO algorithm for a single replication and returns the best objective value.
    """

    y_best, x_best = HACO(objective_function, lb, ub, var_types, population_size,
                     probability_random_chicks, max_step_size, min_step_size,
                     max_iterations, max_runtime, plot_results=False, verbose=False).run()
    return y_best

# Run HACO in parallel for multiple replications
obj_values = Parallel(n_jobs=num_cores)(delayed(run_haco)() for _ in range(num_replications))

# Calculate mean and standard deviation of the objective values
mean_obj_value = stat.mean(obj_values)
std_dev_obj_value = stat.stdev(obj_values)

print(f"Mean objective value: {mean_obj_value}")
print(f"Standard deviation: {std_dev_obj_value}")

