import numpy as np
import cvxpy as cp

def half_thresholding(x, tau):
    """
    Apply L1/2 half-thresholding operator elementwise.
    tau = lambda * mu
    """
    out = np.zeros_like(x)

    threshold = (3 * (tau) ** (2/3)) / (2 ** (4/3))

    for i in range(len(x)):
        xi = x[i]
        if abs(xi) > threshold:
            phi = np.arccos(
                (tau / 8) * (abs(xi) / 3) ** (-3/2)
            )
            out[i] = (
                2/3 * xi *
                (1 + np.cos((2*np.pi/3) - (2/3)*phi))
            )
        else:
            out[i] = 0.0

    return out


def l12_hybrid_index_tracking(train_X, train_y, K,
                               max_iter=200,
                               epsilon=1e-6):

    R = train_X.values
    y = train_y.values.ravel()
    T, N = R.shape
    spectral_norm = np.linalg.norm(R, 2)
    mu = (1 - 1e-4) / (spectral_norm ** 2)
    w = np.zeros(N)
    lambda_val = 1.0
    for _ in range(max_iter):
        gradient_step = w + mu * R.T @ (y - R @ w)
        abs_sorted = np.sort(np.abs(gradient_step))[::-1]
        if K < len(abs_sorted):
            lambda_candidate = (
                np.sqrt(96) / 9 *
                (spectral_norm ** 2) *
                abs_sorted[K] ** (3/2)
            )
            lambda_val = min(lambda_val, lambda_candidate)

        tau = lambda_val * mu
        w_new = half_thresholding(gradient_step, tau)

        if np.linalg.norm(w_new - w) < epsilon:
            break

        w = w_new
        
    support = np.argsort(-np.abs(w))[:K]

    R_subset = R[:, support]

    w_sub = cp.Variable(K)

    objective = cp.Minimize(cp.sum_squares(R_subset @ w_sub - y))
    constraints = [
        cp.sum(w_sub) == 1,
        w_sub >= 0
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    full_w = np.zeros(N)
    full_w[support] = w_sub.value

    return full_w