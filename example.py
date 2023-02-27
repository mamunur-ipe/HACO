import numpy as np
from HACO import HACO
import statistics as stat
from joblib import Parallel, delayed
import multiprocessing as mp

num_cores = mp.cpu_count()


# Objective function
#ACKLEY FUNCTION; obj value:0  Soln:[0, 0]
d = 10 # dimension of the problem
def objective_function(xx):
    a = 20
    b = 0.2
    c = 2*np.pi

    sum1 = 0
    sum2 = 0
    for ii in range(d):
        xi = np.array(xx[ii])
        sum1 = sum1 + xi * xi
        sum2 = sum2 + np.cos(c*xi)

    term1 = -a*np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)
    y = term1 + term2 + a + np.exp(1)
    return y

lb = [-5] * d
ub = [5] * d


#%%
# Set parameter values for HACO algorithm
population = 10 * d
probability_random_chicks = 0.01
max_step = 0.9
min_step = 0.01
max_iteration = 20 * d - 50
max_run_time = 3000


REPLICATIONS = 50
obj_values = []

def find_value(_):
    y_best, x_best = HACO(objective_function, lb, ub, population, probability_random_chicks, max_step, min_step, max_iteration, max_run_time)
    return y_best


# parallel processing
obj_values = Parallel(n_jobs=num_cores)(delayed(find_value)(_) for _ in range(REPLICATIONS))

mean = stat.mean(obj_values)
std_dev = stat.stdev(obj_values)

print(f'Mean value: {mean}')
print(f'Std. Dev.: {std_dev}')

