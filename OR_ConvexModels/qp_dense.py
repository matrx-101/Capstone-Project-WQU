import cvxpy as cp
import numpy as np

def least_squares_long_only(train_X, train_y):
    R = train_X.values
    y = train_y.values.ravel()
    n = R.shape[1]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(R @ w - y))
    constraints = [
        cp.sum(w) == 1,  
        w >= 0      
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return w.value