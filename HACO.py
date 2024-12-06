import numpy as np
import matplotlib.pyplot as plt
import time

# Function to calculate the step size based on iteration
def step_size(max_step, min_step, max_iteration, current_iteration):
    # Linearly decreases the step size from max_step to min_step over iterations
    y = max_step + (min_step - max_step) / max_iteration * current_iteration
    return y

# Function to find the best solution and its fitness value from the current population
def find_best(X, Y):
    idx_best = np.argmin(Y)  # Index of the minimum fitness value
    x_best = X[idx_best]     # Best solution
    y_best = Y[idx_best]     # Best fitness value
    return x_best, y_best

# Main HACO (Hen and Chicks Optimization) algorithm
def HACO(objective_function, lb, ub, var_types, population=30, probability_random_chicks=0.1,
         max_step=0.5, min_step=0.01, max_iteration=500, max_run_time=300):
    """
    Parameters:
    - objective_function: Function to minimize
    - lb: Lower bounds for decision variables
    - ub: Upper bounds for decision variables
    - var_types: List indicating the type of each variable ('float' or 'int')
    - population: Number of chicks in the population
    - probability_random_chicks: Probability of using random search
    - max_step, min_step: Maximum and minimum step sizes
    - max_iteration: Maximum number of iterations
    - max_run_time: Maximum allowable runtime in seconds
    """
    start_time = time.process_time()  # Start the timer

    # Number of decision variables
    no_variables = len(lb)

    # Convert bounds to numpy arrays for easier computation
    lb, ub = np.array(lb), np.array(ub)

    # Step 1: Initialize population and evaluate fitness
    X = np.zeros([population, no_variables])  # Population solutions
    Y = np.zeros(population)  # Fitness values

    for i in range(population):
        for j in range(no_variables):
            if var_types[j] == 'float':
                X[i, j] = lb[j] + np.random.uniform(0, 1) * (ub[j] - lb[j])  # Random initialization for floats
            elif var_types[j] == 'int':
                X[i, j] = np.random.randint(lb[j], ub[j] + 1)  # Random initialization for integers
        Y[i] = objective_function(X[i, :])  # Evaluate fitness

    # Find the initial best solution
    x_best, y_best = find_best(X, Y)

    # History to track the best objective value across iterations
    history_best_obj_values = []

    # Main optimization loop
    for k in range(max_iteration):
        current_step = step_size(max_step, min_step, max_iteration, k)  # Update step size

        # Step 2: Update the positions of the chicks guided by the Hen
        X1 = np.zeros([population, no_variables])
        Y1 = np.zeros(population)
        for i in range(population):
            for j in range(no_variables):
                if var_types[j] == 'float':
                    X1[i, j] = X[i, j] + current_step * np.random.uniform(-1, 1) * (x_best[j] - X[i, j])
                elif var_types[j] == 'int':
                    X1[i, j] = X[i, j] + current_step * np.random.uniform(-1, 1) * (x_best[j] - X[i, j])
                    X1[i, j] = np.round(X1[i, j])  # Round to nearest integer for integer variables

            # Enforce boundary constraints
            if var_types[j] == 'float':
                X1[i, j] = np.clip(X1[i, j], lb[j], ub[j])  # Clipping for float variables
            elif var_types[j] == 'int':
                X1[i, j] = np.clip(np.round(X1[i, j]), lb[j], ub[j])  # Clip and round for integer variables

        # Evaluate fitness for updated positions
        for i in range(population):
            Y1[i] = objective_function(X1[i, :])

        # Step 3: Perform greedy selection (keep better solutions)
        mask = Y1 < Y
        Y[mask] = Y1[mask]
        X[mask, :] = X1[mask, :]

        # Step 4:Update positions of chicks guided by peers or random exploration
        X2 = np.zeros([population, no_variables])
        Y2 = np.zeros(population)
        for i in range(population):
            if np.random.random() >= probability_random_chicks:  # Guided by their peers
                # Select a partner other than self and the best peer
                while True:
                    idx = np.random.randint(0, population)
                    if idx != np.argmin(Y) and idx != i:
                        break
                for j in range(no_variables):
                    if var_types[j] == 'float':
                        X2[i, j] = X[i, j] + current_step * np.random.uniform(-1, 1) * (X[i, j] - X[idx, j])
                    elif var_types[j] == 'int':
                        X2[i, j] = X[i, j] + current_step * np.random.uniform(-1, 1) * (X[i, j] - X[idx, j])
                        X2[i, j] = np.round(X2[i, j])  # Round to nearest integer for integer variables

            else:  # Random chicks (exploration)
                for j in range(no_variables):
                    if var_types[j] == 'float':
                        X2[i, j] = lb[j] + np.random.uniform(0, 1) * (ub[j] - lb[j])
                    elif var_types[j] == 'int':
                        X2[i, j] = np.random.randint(lb[j], ub[j] + 1)

            # Enforce boundary constraints
            for j in range(no_variables):
                if var_types[j] == 'float':
                    X2[i, j] = np.clip(X2[i, j], lb[j], ub[j])  # Clipping for float variables
                elif var_types[j] == 'int':
                    X2[i, j] = np.clip(np.round(X2[i, j]), lb[j], ub[j])  # Clip and round for integer variables

        # Evaluate fitness for updated positions of the chicks
        for i in range(population):
            Y2[i] = objective_function(X2[i, :])

        # Step 6: Perform greedy selection
        mask = Y2 < Y
        Y[mask] = Y2[mask]
        X[mask, :] = X2[mask, :]

        # Update the best solution
        x_best, y_best = find_best(X, Y)

        # Record the best objective value for this iteration
        history_best_obj_values.append(y_best)

        # print the current result
        print(f'Iteration:{k+1} Solution: {x_best} Objective: {np.min(Y)}')

        # Break the loop if maximum runtime is exceeded
        current_time = time.process_time()
        if (current_time - start_time) > max_run_time:
            break

    # create plot
    plt.figure(dpi = 300, figsize =(6, 4), constrained_layout=True )
    plt.plot( range(1, len(history_best_obj_values)+1), history_best_obj_values, color='r')
    plt.xlabel('Iterations')
    plt.ylabel('Objective function value')
    plt.show()

    return y_best, x_best

# Example usage with the Ackley function (using both float and integer variables)
if __name__ == '__main__':
    # Ackley Function
    d = 4  # Dimension of the function
    def objective_function(xx):
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum1 = sum([xi ** 2 for xi in xx])
        sum2 = sum([np.cos(c * xi) for xi in xx])
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)

    # Bounds for variables
    lb = [-5]*d  # First variable float, second variable int
    ub = [5]*d  # First variable float, second variable int

    # Variable types (float for the first two variable, int for the last two variables)
    var_types = ['float', 'float', 'int', 'int']

    # Algorithm parameters
    population = 30
    probability_random_chicks = 0.1
    max_step = 0.99
    min_step = 0.01
    max_iteration = 100
    max_run_time = 300

    # Run the HACO algorithm
    y_best, x_best = HACO(objective_function, lb, ub, var_types, population, probability_random_chicks, max_step, min_step, max_iteration, max_run_time)

    # Print the results
    print(f'Best objective value: {y_best}')
    print(f'Best solution: {x_best}')
