import numpy as np
import cvxpy as cp

def least_squares_ridge_long_only(
    train_X,
    train_y,
    lambda_l2=1e-3,
    solver="OSQP"
):
    R = train_X.values
    y = train_y.values.ravel()

    T, n = R.shape
    R_scaled = R / np.sqrt(T)
    y_scaled = y / np.sqrt(T)
    w = cp.Variable(n)
    objective = cp.Minimize(
        cp.sum_squares(R_scaled @ w - y_scaled)
        + lambda_l2 * cp.sum_squares(w)
    )
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(
            solver=solver,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=200000,
            verbose=False
        )
    except:
        problem.solve(verbose=False)

    if w.value is None:
        raise ValueError("Optimization failed. Try different lambda or solver.")
    w_opt = np.maximum(w.value, 0)
    w_opt = w_opt / w_opt.sum()

    return w_opt