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


def genetic_algo_sparse_hybrid(train_X, train_y, K,
                               w_dense,
                               population_size=50,
                               generations=50,
                               mutation_rate=0.1):

    R = train_X.values
    y = train_y.values.ravel()
    N = R.shape[1]

    # Dense-based probability with smoothing
    prob = np.abs(w_dense)

    if prob.sum() == 0:
        prob = np.ones(N) / N
    else:
        prob = prob / prob.sum()

    # small smoothing to avoid collapse
    epsilon = 0.05
    prob = (1 - epsilon) * prob + epsilon / N
    prob = prob / prob.sum()

    ranked_indices = np.argsort(-prob)

    # Initialize Population
    population = []

    # Deterministic Top-K
    top_k = ranked_indices[:K].tolist()
    population.append(top_k)

    # Perturbed Top-K
    for _ in range(population_size // 5):
        base = top_k.copy()
        swap_idx = random.randrange(K)

        remaining = list(set(range(N)) - set(base))
        if len(remaining) > 0:
            remaining_prob = prob[remaining]
            remaining_prob /= remaining_prob.sum()
            candidate = int(np.random.choice(remaining, p=remaining_prob))
            base[swap_idx] = candidate

        population.append(base)

    # Probability-weighted sampling
    for _ in range(population_size // 2):
        subset = np.random.choice(N, K, replace=False, p=prob)
        population.append(subset.tolist())

    # Pure random for diversity
    while len(population) < population_size:
        population.append(random.sample(range(N), K))

    # ---- Fitness Function ----
    def fitness(chromosome):
        R_subset = R[:, chromosome]
        w = solve_subset_weights(R_subset, y)
        if w is None:
            return 1e10
        pred = R_subset @ w
        return np.mean((pred - y) ** 2)

    # ---- Evolution ----
    for _ in range(generations):

        scores = [(chrom, fitness(chrom)) for chrom in population]
        scores.sort(key=lambda x: x[1])

        elite_size = population_size // 5
        new_population = [scores[i][0] for i in range(elite_size)]

        while len(new_population) < population_size:

            parent1 = random.choice(scores[:elite_size])[0]
            parent2 = random.choice(scores[:elite_size])[0]

            half = K // 2
            child = parent1[:half] + parent2[half:]

            # remove duplicates
            seen = set()
            child = [x for x in child if not (x in seen or seen.add(x))]

            # fill from remaining pool safely
            remaining = list(set(range(N)) - set(child))

            if len(remaining) >= K - len(child):
                remaining_prob = prob[remaining]
                remaining_prob /= remaining_prob.sum()
                chosen = np.random.choice(
                    remaining,
                    size=K - len(child),
                    replace=False,
                    p=remaining_prob
                )
                child.extend(chosen.tolist())
            else:
                # fallback uniform sampling
                needed = K - len(child)
                child.extend(random.sample(remaining, needed))

            # mutation
            if random.random() < mutation_rate:
                remaining = list(set(range(N)) - set(child))
                if len(remaining) > 0:
                    idx = random.randrange(K)
                    candidate = random.choice(remaining)
                    child[idx] = candidate

            new_population.append(child[:K])

        population = new_population

    # ---- Final Selection ----
    best_chrom = min(population, key=fitness)

    R_subset = R[:, best_chrom]
    w_subset = solve_subset_weights(R_subset, y)

    full_w = np.zeros(N)
    for i, idx in enumerate(best_chrom):
        full_w[idx] = w_subset[i]

    return full_w