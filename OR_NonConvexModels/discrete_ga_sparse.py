import numpy as np
import cvxpy as cp
import random

# Solve long-only QP for a fixed subset
def solve_subset_weights(R_subset, y):
    T, K = R_subset.shape

    # scale for numerical stability
    R_scaled = R_subset / np.sqrt(T)
    y_scaled = y / np.sqrt(T)

    w = cp.Variable(K)

    objective = cp.Minimize(cp.sum_squares(R_scaled @ w - y_scaled))
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.OSQP,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=100000,
        verbose=False
    )

    if w.value is None:
        return None

    return w.value


# Genetic Algorithm Sparse Selection
def genetic_algo_sparse(train_X,
                        train_y,
                        K,
                        population_size=50,
                        generations=50,
                        mutation_rate=0.1,
                        tournament_size=3,
                        random_state=42):

    random.seed(random_state)
    np.random.seed(random_state)

    R = train_X.values
    y = train_y.values.ravel()
    T, N = R.shape

    assert K < T, "K must be smaller than training sample size T."

    # ----------------------------
    # Fitness cache (important)
    # ----------------------------
    fitness_cache = {}

    def fitness(chromosome):
        key = tuple(sorted(chromosome))
        if key in fitness_cache:
            return fitness_cache[key]

        R_subset = R[:, chromosome]
        w = solve_subset_weights(R_subset, y)

        if w is None:
            score = 1e10
        else:
            pred = R_subset @ w
            score = np.mean((pred - y) ** 2)

        fitness_cache[key] = score
        return score

    # Initialize population
    population = [
        random.sample(range(N), K)
        for _ in range(population_size)
    ]

    # Tournament selection
    def tournament_select():
        candidates = random.sample(population, tournament_size)
        candidates.sort(key=lambda c: fitness(c))
        return candidates[0]

    # Crossover
    def crossover(parent1, parent2):
        union = list(set(parent1) | set(parent2))

        if len(union) >= K:
            return random.sample(union, K)
        else:
            remaining = list(set(range(N)) - set(union))
            return union + random.sample(remaining, K - len(union))

    # Mutation
    def mutate(chromosome):
        chrom = chromosome.copy()
        if random.random() < mutation_rate:
            idx_to_replace = random.randrange(K)
            available = list(set(range(N)) - set(chrom))
            if available:
                chrom[idx_to_replace] = random.choice(available)
        return chrom

    # Evolution loop
    for _ in range(generations):

        population.sort(key=lambda c: fitness(c))

        elite_size = population_size // 5
        new_population = population[:elite_size]  # elitism

        while len(new_population) < population_size:
            parent1 = tournament_select()
            parent2 = tournament_select()

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    # Select best solution
    best_chrom = min(population, key=lambda c: fitness(c))

    R_subset = R[:, best_chrom]
    w_subset = solve_subset_weights(R_subset, y)

    full_w = np.zeros(N)
    for i, idx in enumerate(best_chrom):
        full_w[idx] = w_subset[i]

    return full_w