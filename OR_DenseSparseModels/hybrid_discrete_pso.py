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


def hybrid_pso_sparse(train_X, train_y, K,
                      w_dense,
                      swarm_size=30,
                      iterations=50,
                      c1=0.4,
                      c2=0.4):

    R = train_X.values
    y = train_y.values.ravel()
    N = R.shape[1]

    # Dense probability with smoothing
    prob = np.abs(w_dense)
    if prob.sum() == 0:
        prob = np.ones(N) / N
    else:
        prob = prob / prob.sum()

    epsilon = 0.05
    prob = (1 - epsilon) * prob + epsilon / N
    prob = prob / prob.sum()

    ranked_indices = np.argsort(-prob)

    # Initialize swarm
    swarm = []

    top_k = ranked_indices[:K].tolist()
    swarm.append(top_k)

    for _ in range(swarm_size // 4):
        base = top_k.copy()
        swap_idx = random.randrange(K)

        remaining = list(set(range(N)) - set(base))
        if remaining:
            remaining_prob = prob[remaining]
            remaining_prob /= remaining_prob.sum()
            candidate = int(np.random.choice(remaining, p=remaining_prob))
            base[swap_idx] = candidate

        swarm.append(base)

    for _ in range(swarm_size // 2):
        subset = np.random.choice(N, K, replace=False, p=prob)
        swarm.append(subset.tolist())

    while len(swarm) < swarm_size:
        swarm.append(random.sample(range(N), K))

    # ----- Fitness -----
    def fitness(subset):
        R_subset = R[:, subset]
        w = solve_subset_weights(R_subset, y)
        if w is None:
            return 1e10
        pred = R_subset @ w
        return np.mean((pred - y) ** 2)

    pbest = swarm.copy()
    pbest_scores = [fitness(p) for p in swarm]

    gbest_idx = np.argmin(pbest_scores)
    gbest = pbest[gbest_idx]
    gbest_score = pbest_scores[gbest_idx]

    # Iterations
    for _ in range(iterations):

        for i in range(swarm_size):

            new_particle = swarm[i].copy()

            # move toward pbest
            for j in range(K):
                if random.random() < c1:
                    new_particle[j] = pbest[i][j]

            # move toward gbest
            for j in range(K):
                if random.random() < c2:
                    new_particle[j] = gbest[j]

            # remove duplicates
            seen = set()
            new_particle = [x for x in new_particle if not (x in seen or seen.add(x))]

            # SAFE FILL
            remaining = list(set(range(N)) - set(new_particle))

            if len(remaining) >= K - len(new_particle):
                remaining_prob = prob[remaining]
                remaining_prob /= remaining_prob.sum()
                chosen = np.random.choice(
                    remaining,
                    size=K - len(new_particle),
                    replace=False,
                    p=remaining_prob
                )
                new_particle.extend(chosen.tolist())
            else:
                needed = K - len(new_particle)
                new_particle.extend(random.sample(remaining, needed))

            new_particle = new_particle[:K]

            swarm[i] = new_particle

            score = fitness(new_particle)

            if score < pbest_scores[i]:
                pbest[i] = new_particle
                pbest_scores[i] = score

        best_idx = np.argmin(pbest_scores)
        if pbest_scores[best_idx] < gbest_score:
            gbest = pbest[best_idx]
            gbest_score = pbest_scores[best_idx]

    # Final weights
    R_subset = R[:, gbest]
    w_subset = solve_subset_weights(R_subset, y)

    full_w = np.zeros(N)
    for i, idx in enumerate(gbest):
        full_w[idx] = w_subset[i]

    return full_w