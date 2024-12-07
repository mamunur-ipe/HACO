'''
Author: Mamunur Rahman
Email: mamunur.ipe05@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import warnings

# Chicken class to represent each individual in the population
class Chicken:
    def __init__(self, position, fitness, is_hen=False):
        self.position = position  # Position (solution) of the chicken
        self.fitness = fitness  # Fitness value of the chicken
        self.is_hen = is_hen  # Flag indicating if the chicken is the Hen

    def update_position(self, new_position):
        """Update the chicken's position."""
        self.position = new_position

    def update_fitness(self, new_fitness):
        """Update the chicken's fitness."""
        self.fitness = new_fitness

    def set_as_hen(self):
        """Set the chicken as the Hen (leader)."""
        self.is_hen = True


# HACO algorithm class implementing Hen and Chicks Optimization
class HACO:
    def __init__(self, objective_function, lb, ub, var_types, population=30, probability_random_chicks=0.1,
                 max_step=0.5, min_step=0.01, max_iteration=500, max_run_time=2*3600,
                 plot_results=True, verbose=False, progress_bar = True):
        """
        Initialize the HACO algorithm parameters.

        Parameters:
        - objective_function: The function to minimize.
        - lb: Lower bounds for the variables.
        - ub: Upper bounds for the variables.
        - var_types: Types of the variables. Example: ['float', 'int']
        - population: Number of chickens in the population.
        - probability_random_chicks: Probability of random exploration for chicks.
        - max_step: Maximum step size for position updates.
        - min_step: Minimum step size for position updates.
        - max_iteration: Maximum number of iterations.
        - max_run_time: Maximum runtime in seconds.
        - plot_results: Whether to plot the objective function values over iterations.
        - verbose: Whether to print iteration details.
        - progress_bar: Whether to show the progress bar. Verbose need to be disabled to show the progress bar.
        """
        self.objective_function = objective_function
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.var_types = var_types
        self.population = population
        self.probability_random_chicks = probability_random_chicks
        self.max_step = max_step
        self.min_step = min_step
        self.max_iteration = max_iteration
        self.max_run_time = max_run_time
        self.plot_results = plot_results
        self.verbose = verbose
        self.progress_bar = progress_bar
        self.history_best_obj_values = []

    def step_size(self, current_iteration):
        """
        Calculate the step size based on the current iteration.
        Linearly decreases from max_step to min_step over iterations.
        """
        return self.max_step + (self.min_step - self.max_step) / self.max_iteration * current_iteration

    def find_best(self, chickens):
        """
        Identify the chicken with the best (minimum) fitness value in a population of chickens.
        Returns the position and fitness of the best chicken.
        """
        best_chicken = min(chickens, key=lambda x: x.fitness)
        return best_chicken.position, best_chicken.fitness

    def initialize_population(self):
        """
        Initialize the chicken population with random positions and compute fitness values.
        """
        chickens = []
        for _ in range(self.population):
            position = []
            for lb_i, ub_i, var_type in zip(self.lb, self.ub, self.var_types):
                if var_type == 'float':
                    value = np.random.uniform(lb_i, ub_i)
                elif var_type == 'int':
                    value = np.random.randint(lb_i, ub_i + 1)
                else:
                    raise ValueError("Variable type must be 'float' or 'int'")
                position.append(value)

            position = np.array(position)
            fitness = self.objective_function(position)
            chickens.append(Chicken(position, fitness))
        return chickens

    def enforce_variable_types(self, position):
        """
        Ensure variables conform to their specified types ('int' or 'float').
        """
        for i, var_type in enumerate(self.var_types):
            if var_type == 'int':
                position[i] = round(position[i])
        return position

    def show_progress(self, count, total):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percent = round(100.0 * count / total, 1)
        bar = 'RUNNING: ' + '|' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(f'\r{bar} {percent}% complete')
        sys.stdout.flush()
        # time.sleep(0.001)

    def run(self):
        """
        Run the HACO optimization algorithm.
        Returns:
        - Best fitness value found.
        - Best solution found.
        """
        start_time = time.process_time()
        chickens = self.initialize_population()

        # Find the initial best solution and assign the hen in that location
        x_best, y_best = self.find_best(chickens)
        best_chicken = Chicken(x_best, y_best)
        best_chicken.set_as_hen()

        for k in range(self.max_iteration):

            # Calculate the step size for the current iteration
            current_step = self.step_size(current_iteration = k)

            # Update the positions of the chicks guided by the Hen
            new_chickens = []
            for chicken in chickens:
                if chicken.is_hen:
                    new_chickens.append(chicken)
                    continue
                new_position = chicken.position + current_step * np.random.uniform(-1, 1, len(chicken.position)) * (best_chicken.position - chicken.position)
                new_position = np.clip(new_position, self.lb, self.ub)
                new_position = self.enforce_variable_types(new_position)
                new_fitness = self.objective_function(new_position)
                new_chickens.append(Chicken(new_position, new_fitness))

            # Perform greedy selection (keep better solutions)
            for i, chicken in enumerate(new_chickens):
                if chicken.fitness < chickens[i].fitness:
                    chickens[i].update_position(chicken.position)
                    chickens[i].update_fitness(chicken.fitness)

            # Update positions of the chicks guided by their peers or random exploration
            new_chickens_peer_guided = []
            for i, chicken in enumerate(chickens):
                if np.random.random() >= self.probability_random_chicks:
                    while True:
                        idx = np.random.randint(0, self.population)
                        if idx != np.argmin([c.fitness for c in chickens]) and idx != i:
                            break
                    new_position = chicken.position + current_step * np.random.uniform(-1, 1, len(chicken.position)) * (chicken.position - chickens[idx].position)
                else:
                    new_position = np.array([
                        np.random.uniform(lb_i, ub_i) if var_type == "float" else np.random.randint(lb_i, ub_i + 1)
                        for lb_i, ub_i, var_type in zip(self.lb, self.ub, self.var_types)
                    ])
                new_position = np.clip(new_position, self.lb, self.ub)
                new_position = self.enforce_variable_types(new_position)
                new_fitness = self.objective_function(new_position)
                new_chickens_peer_guided.append(Chicken(new_position, new_fitness))

            # Perform greedy selection for peer-guided chicks
            for i, chicken in enumerate(new_chickens_peer_guided):
                if chicken.fitness < chickens[i].fitness:
                    chickens[i].update_position(chicken.position)
                    chickens[i].update_fitness(chicken.fitness)

            # Update the best solution
            x_best, y_best = self.find_best(chickens)
            best_chicken.update_position(x_best)
            best_chicken.update_fitness(y_best)

            self.history_best_obj_values.append(best_chicken.fitness)

            if self.verbose:  # Print iteration details if verbose is True
                print(f'Iteration:{k + 1} Objective: {best_chicken.fitness}')

            # show progress bar
            if self.progress_bar and (self.verbose==False):
                self.show_progress(k+1, self.max_iteration)

            # Terminate the algorithm if runtime limit exceeds
            current_time = time.process_time()
            if (current_time - start_time) > self.max_run_time:
                warnings.warn("The algorithm terminated early because the provided maximum runtime limit was exceeded!", UserWarning)
                break

        if self.plot_results:  # Plot results if plot_results is True
            plt.figure(dpi=300, figsize=(6, 4), constrained_layout=True)
            plt.plot(range(1, len(self.history_best_obj_values) + 1), self.history_best_obj_values, color='r')
            plt.xlabel('Iterations')
            plt.ylabel('Objective function value')
            plt.show()

        return best_chicken.fitness, best_chicken.position


if __name__ == '__main__':
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
    var_types = ['float', 'float', 'float', 'float', 'int', 'int', 'int', 'int']  # First 4 variables are floats, last 4 are integers

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
    
