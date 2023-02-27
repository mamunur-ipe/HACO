#ACKLEY FUNCTION; obj value:0  Soln:[0, 0]
d = 5 # dimension of the problem
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
# SPHERE FUNCTION
d = 5
def objective_function(xx):
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        sum += xi ** 2

    y = sum
    return y

lb = [-5] * d
ub = [5] * d

#%%
# Rotated Hyper-ellipsoid
d = 5
def objective_function(xx):
    outer = 0
    for ii in range(d):
        inner = 0
        for jj in range(ii):
            xj = xx[jj]
            inner += xj ** 2
        outer += inner

    y = outer
    return y

lb = [-5] * d
ub = [5] * d


#%%
#GRIEWANK FUNCTION
d = 5
def objective_function(xx):
    s = 0
    p = 1

    for ii in range(d):
        xi = xx[ii]
        s = s + xi**2/4000
        p = p * np.cos(xi/np.sqrt(ii+1))

    y = s - p + 1
    return y

#%%
# SCHWEFEL FUNCTION

d = 5
def objective_function(xx):
    s = 0

    for ii in range(d):
        xi = xx[ii]
        s = s + xi * np.sin(np.sqrt(np.abs(xi)))

    y = 418.9829 * d - s
    return y

#%%
# HappyCat function
d = 5
def objective_function(xx):
    sum1 = 0
    sum2 = 0
    for ii in range(d):
        xi = xx[ii]
        sum1 +=xi
        sum2 += xi**2
    y = (abs(sum2 - d))**0.25 + (0.5*sum2 + sum1)/d + 0.5
    return y

lb = [-20] * d
ub = [20] * d

#%%
# HGBat function
d = 5
def objective_function(xx):
    sum1 = 0
    sum2 = 0
    for ii in range(d):
        xi = xx[ii]
        sum1 +=xi
        sum2 += xi**2
    y = (abs(sum2**2 - sum1**2))**0.5 + (0.5*sum2 + sum1)/d + 0.5
    return y

lb = [-15] * d
ub = [15] * d


#%%
# Ellipsoid
d = 5
def objective_function(xx):
    sum1 = 0
    for ii in range(d):
        xi = xx[ii]
        sum1 += (ii+1)*xi**2
        y = sum1
    return y

lb = [-100] * d
ub = [100] * d



#%%
# Sum of Dif. Powers 2
d = 5
def objective_function(xx):
    sum1 = 0
    for ii in range(d):
        i = ii + 1
        xi = xx[ii]
        power = 2 + (4*(i - 1)/(d - 1))
        sum1 += np.abs(xi)**power
    y = sum1

    return y

lb = [-10] * d
ub = [10] * d

#%%
# Sum of Different Powers function
d = 5
def objective_function(xx):
    sum1 = 0
    for ii in range(d):
        xi = xx[ii]
        i = ii + 1
        sum1 += np.abs(xi)**(i + 1)
        y = sum1
    return y

lb = [-10] * d
ub = [10] * d

#%%
# Quintic Function
d = 5
def objective_function(xx):
    sum1 = 0
    for ii in range(d):
        xi = xx[ii]
        sum1 += np.abs(xi**5 -3*xi**4 + 4*xi**3 + 2*xi**2 - 10*xi -4)
        y = sum1
    return y

lb = [-20] * d
ub = [20] * d

#%%
# Weierstrass function

d = 5
def objective_function(xx):
    a = 0.5
    b = 3
    k_max = 20
    k_vec = np.arange(k_max + 1)
    sum2 = np.sum(a ** k_vec * np.cos(np.pi * b ** k_vec))

    f = 0
    for i in range(d):
        xi = xx[i]
        sum1 = np.sum(a ** k_vec * np.cos(2 * np.pi * b ** k_vec * (xi + 0.5)))
        f += sum1

    y = f - d * sum2
    return y

lb = [-0.5] * d
ub = [0.5] * d

#%%
# Alpine 1 Function
d = 5
def objective_function(xx):
    sum1 = 0
    for ii in range(d):
        xi = xx[ii]
        sum1 += np.abs( xi * np.sin(xi) + 0.1 * xi )
        y = sum1
    return y

lb = [-10] * d
ub = [10] * d
