from cvxpy import *
import numpy as np
from posdef import nearestPD


def MarkowitzOpt(mean, variance, covariance, interest_rate, min_return):
    n = mean.size + 1                   # Number of assets (number of stocks + interest rate)
    
    mu = mean.values                    # Mean returns of n assets
    temp = np.full(n, interest_rate)
    temp[:-1] = mu
    mu = temp
        
    counter = 0
    Sigma = np.zeros((n,n))                 # Covariance of n assets
    for i in np.arange(n-1):
        for j in np.arange(i, n-1):
            if i==j:
                Sigma[i,j] = variance[i]
            else:
                Sigma[i,j] = covariance[counter]
                Sigma[j,i] = Sigma[i,j]
                counter+=1
    Sigma = nearestPD(Sigma)                # Converting covariance to the nearest positive-definite matrix
    
    # Ensuring feasability of inequality contraint
    if mu.max() < min_return:
        min_return = interest_rate
    
    w = Variable(n)                         # Portfolio allocation vector
    ret = mu.T*	w
    risk = quad_form(w, Sigma)
    min_ret = Parameter(sign='positive')
    min_ret.value = min_return
    prob = Problem(Minimize(risk),          # Restricting to long-only portfolio
                   [ret >= min_ret,
                   sum_entries(w) == 1,
                   w >= 0])
    prob.solve()
    return w.value