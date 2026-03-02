import pandas as pd
import numpy as np
import cvxpy as cp
import random

def solve_subset_weights(R_subset, y):
    T, K = R_subset.shape
    w = cp.Variable(K)
    objective = cp.Minimize(cp.sum_squares(R_subset @ w - y))
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        return None
    return w.value

def pso_sparse(train_X, train_y, K,
               swarm_size=30,
               iterations=50):

    R = train_X.values
    y = train_y.values.ravel()
    N = R.shape[1]

    def fitness(subset):
        R_subset = R[:, subset]
        w = solve_subset_weights(R_subset, y)
        if w is None:
            return 1e10
        pred = R_subset @ w
        return np.mean((pred - y) ** 2)

    # Initialize swarm
    swarm = [random.sample(range(N), K) for _ in range(swarm_size)]
    pbest = swarm.copy()
    pbest_scores = [fitness(p) for p in swarm]

    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = min(pbest_scores)

    # Iterations
    for _ in range(iterations):

        for i in range(swarm_size):

            current = swarm[i]
            new_particle = current.copy()

            # Move toward personal best
            for idx in range(K):
                if random.random() < 0.3:
                    new_particle[idx] = pbest[i][idx]

            # Move toward global best
            for idx in range(K):
                if random.random() < 0.3:
                    new_particle[idx] = gbest[idx]

            # Remove duplicates
            new_particle = list(dict.fromkeys(new_particle))

            # Fill if needed
            while len(new_particle) < K:
                candidate = random.randrange(N)
                if candidate not in new_particle:
                    new_particle.append(candidate)

            new_particle = new_particle[:K]

            swarm[i] = new_particle

            score = fitness(new_particle)

            # Updating the personal best
            if score < pbest_scores[i]:
                pbest[i] = new_particle
                pbest_scores[i] = score

        # Updating the global best
        best_idx = np.argmin(pbest_scores)
        if pbest_scores[best_idx] < gbest_score:
            gbest = pbest[best_idx]
            gbest_score = pbest_scores[best_idx]

    # the final weights
    R_subset = R[:, gbest]
    w_subset = solve_subset_weights(R_subset, y)

    full_w = np.zeros(N)
    for i, idx in enumerate(gbest):
        full_w[idx] = w_subset[i]

    return full_w
