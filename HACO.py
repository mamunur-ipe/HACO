import numpy as np
import matplotlib.pyplot as plt
import time


def step_size(max_step, min_step, max_iteration, current_iteration):
    y = max_step + (min_step - max_step)/max_iteration * current_iteration
    return y

# locate the best solution and fitness value: x_best, y_best
def find_best(X, Y):
    idx_best = np.argmin(Y)
    x_best = X[idx_best]
    y_best = Y[idx_best]
    return x_best, y_best

# define the main function
def HACO(objective_function, lb, ub, population=30, probability_random_chicks=0.1, max_step=0.5, min_step=0.01, max_iteration=500, max_run_time=300):
    start_time = time.process_time()    #track execution time
    # step 1: initial solution
    no_variables = len(lb)
    # convert to numpy array
    lb, ub = np.array(lb), np.array(ub)

    X = np.zeros([population, no_variables])
    Y = np.zeros(population) # store fitness value

    for i in range(population):
        X[i,:] = lb + np.random.uniform(0, 1, no_variables)*(ub - lb)
        Y[i] = objective_function(X[i, :])

    x_best, y_best = find_best(X, Y)

    history_best_obj_values = []  #keep history of the best fitness value for each iteration
    for k in range(max_iteration):
        current_step = step_size(max_step, min_step, max_iteration, k)

        # step 2: Guided by the Hen----------------------------------------------------------------
        X1 = np.zeros([population, no_variables])
        Y1 = np.zeros(population)  # store fitness value
        for i in range(population):

            X1[i,:] = X[i,:] + current_step * np.random.uniform(-1, 1, no_variables) * ( x_best - X[i,:] )
            # X1[i,:] = x_best + current_step*np.random.uniform(-1, 1, no_variables)*( x_best)

            ## keep the search space within bounds
            # check lower bound
            mask = X1[i,:] < lb
            X1[i,:][mask] = lb[mask]
            # check upper bound
            mask = X1[i,:] > ub
            X1[i,:][mask] = ub[mask]

        # store objective function value for all the population
        for i in range(population):
            Y1[i] = objective_function(X1[i,:])

        # step 3: perform greedy selection
        mask = Y1 < Y
        Y[mask] = Y1[mask]
        X[mask, :] = X1[mask, :]



        # step 4: Guided by other chicks--------------------------------------------------------------
        X2 = np.zeros([population, no_variables])
        Y2 = np.zeros(population)  # store fitness value

        for i in range(population):

            if np.random.random() >= probability_random_chicks: #guided by other chicks
                # select a partner other than self and the best group member
                while True:
                    idx =np.random.randint(0, population)
                    if ( idx != np.argmin(Y) ) and (idx != i):
                        break
                X2[i,:] = X[i,:] + current_step * np.random.uniform(-1, 1, no_variables) * (X[i,:] - X[idx, :])

            else: # step 5: random chicks-----------------------
                X2[i,:] = lb + np.random.uniform(0, 1, no_variables)*(ub - lb)

            ## keep the search space within bounds
            # check lower bound
            mask = X2[i,:] < lb
            X2[i,:][mask] = lb[mask]
            # check upper bound
            mask = X2[i,:] > ub
            X2[i,:][mask] = ub[mask]

        # store objective function value for all the population
        for i in range(population):
            Y2[i] = objective_function(X2[i,:])

        # step 6: perform greedy selection
        mask = Y2 < Y
        Y[mask] = Y2[mask]
        X[mask, :] = X2[mask, :]


        # current best solution
        x_best, y_best = find_best(X, Y)
        # current best fitness value
        history_best_obj_values.append( y_best )


        #print the current result
        # print(f'Iteration:{k+1} Solution: {x_best} Objective: {np.min(Y)}')

        ## break the loop if any of the termination conditions is true
        current_time = time.process_time()
        if (current_time - start_time) > max_run_time:  # max_run_time in seconds
            break

    # # print the best result
    # print('\n------------------------------------------------------------')
    # print(f'Elapsed time: {current_time - start_time} seconds')
    # print(f'Best objective value : {y_best}')
    # print(f'Best solution        : {x_best}')

    # # create plot
    # plt.figure(dpi = 300, figsize =(6, 4), constrained_layout=True )
    # plt.plot( range(1, len(history_best_obj_values)+1), history_best_obj_values, color='r')
    # plt.xlabel('Iterations')
    # plt.ylabel('Objective function value')
    # plt.show()


    return y_best, x_best

#===============================================================================

if __name__ == '__main__':

    #ACKLEY FUNCTION; obj value:0  Soln:[0, 0]
    d = 2 # dimension of the problem
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



    # parameters
    population = 30
    probability_random_chicks = 0.0
    max_step = 0.99
    min_step = 0.01
    max_iteration = 200
    max_run_time = 300  # max_run_time in seconds

    y_best, x_best =  HACO(objective_function, lb, ub, population, probability_random_chicks, max_step, min_step, max_iteration, max_run_time)

    print(f'Best objective value: {y_best}')
    print(f'Best solution: {x_best}')

