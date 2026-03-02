import cvxpy as cp
import numpy as np

def l1_sparse_tracking_or(train_X, train_y, lambda_reg=0.01):
    R = train_X.values
    y = train_y.values.ravel()
    n = R.shape[1]

    w = cp.Variable(n)

    objective = cp.Minimize(
        cp.sum_squares(R @ w - y)
        + lambda_reg * cp.norm1(w)
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0  
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    return w.value