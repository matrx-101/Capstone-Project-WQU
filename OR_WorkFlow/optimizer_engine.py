import pandas as pd
import numpy as np
import time
from OR_BaselineModels import (
    equal_weight,
    least_square
)
from OR_ConvexModels import (
    l1_qp_soft_sparse,
    l2_qp_dense,
    qp_dense
)
from OR_NonConvexModels import (
    discrete_ga_sparse,
    discrete_pso_sparse,
    l12_sparse
)
from OR_Dataset.data_engineering import (
    load_OR_index_dataset,
    convert_prices_to_returns,
    train_test_split_or
)
from OR_Metrics import (
    consistency,
    sparsity,
    superiority,
    te_metrics
)
from OR_DenseSparseModels import (
    hybrid_discrete_ga,
    hybrid_discrete_pso
)

def run_or_convex_experiment(index_name,
                             K_values=[5, 10, 15],
                             population_size=50,
                             generations=50,
                             mutation_rate=0.1,
                             swarm_size=30,
                             iterations=50
                             ):
    stocks_df, index_df = load_OR_index_dataset(index_name)
    print(f"Number of Periods {stocks_df.shape[0]}, Number of Assets {stocks_df.shape[1]}")
    stock_returns, index_returns = convert_prices_to_returns(
        stocks_df,
        index_df
    )
    print("Std Deviation Stock Returns",stock_returns.std().mean())
    print("Std Deviation Index Returns",index_returns.std())
    
    train_X, test_X, train_y, test_y = train_test_split_or(
        stock_returns,
        index_returns
    )
    N_assets = train_X.shape[1]
    K_values = K_values
    K_values = [K for K in K_values if K < N_assets]
    
    results = []
    #Equal Weight (Baseline)
    start = time.perf_counter()
    w_eq = equal_weight.equal_weight_or(train_X)
    runtime_eq = time.perf_counter() - start

    TE_I_eq, TE_O_eq = te_metrics.compute_te_metrics(
        train_X, train_y,
        test_X, test_y,
        w_eq
    )

    results.append({
        "Model": "EqualWeight",
        "TE_I": TE_I_eq,
        "TE_O": TE_O_eq,
        "Cons": consistency.consistency(TE_I_eq, TE_O_eq),
        "SupO_EQ": 0.0,
        "Sparsity": sparsity.sparsity(w_eq),
        "Runtime": runtime_eq
    })
    # Least Squares (Baseline)
    start = time.perf_counter()
    w_ls = least_square.least_squares_or(train_X, train_y)
    runtime_ls = time.perf_counter() - start

    TE_I_ls, TE_O_ls = te_metrics.compute_te_metrics(
        train_X, train_y,
        test_X, test_y,
        w_ls
    )

    results.append({
        "Model": "LeastSquares",
        "TE_I": TE_I_ls,
        "TE_O": TE_O_ls,
        "Cons": consistency.consistency(TE_I_ls, TE_O_ls),
        "SupO_EQ": superiority.superiority_oos(TE_O_eq, TE_O_ls),
        "Sparsity": sparsity.sparsity(w_ls),
        "Runtime": runtime_ls
    })
    
    # Long-Only Dense QP
    start = time.perf_counter()
    w_qp = qp_dense.least_squares_long_only(train_X, train_y)
    runtime_qp = time.perf_counter() - start

    TE_I_qp, TE_O_qp = te_metrics.compute_te_metrics(
        train_X, train_y,
        test_X, test_y,
        w_qp
    )

    results.append({
        "Model": "DenseQP",
        "TE_I": TE_I_qp,
        "TE_O": TE_O_qp,
        "Cons": consistency.consistency(TE_I_qp, TE_O_qp),
        "SupO_EQ": superiority.superiority_oos(TE_O_eq, TE_O_qp),
        "Sparsity": sparsity.sparsity(w_qp),
        "Runtime": runtime_qp
    })
    
    # L1 Soft Sparse
    start = time.perf_counter()
    w_l1 = l1_qp_soft_sparse.l1_sparse_tracking_or(train_X, train_y)
    runtime_l1 = time.perf_counter() - start

    TE_I_l1, TE_O_l1 = te_metrics.compute_te_metrics(
        train_X, train_y,
        test_X, test_y,
        w_l1
    )
    results.append({
        "Model": "L1_Soft_Sparse",
        "TE_I": TE_I_l1,
        "TE_O": TE_O_l1,
        "Cons": consistency.consistency(TE_I_l1, TE_O_l1),
        "SupO_EQ": superiority.superiority_oos(TE_O_eq, TE_O_l1),
        "SupO_QP":superiority.superiority_oos(TE_O_qp, TE_O_l1),
        "SupO_L1":0.0,
        "SupO_L2":superiority.superiority_oos(TE_O_l2, TE_O_l1),
        "Sparsity": sparsity.sparsity(w_l1),
        "Runtime": runtime_l1
    })
    
    # L2 Ridge Regularization
    start = time.perf_counter()
    w_l2 = l2_qp_dense.least_squares_ridge_long_only(train_X, train_y)
    runtime_l2 = time.perf_counter() - start

    TE_I_l2, TE_O_l2 = te_metrics.compute_te_metrics(
        train_X, train_y,
        test_X, test_y,
        w_l2
    )
    results.append({
        "Model": "L2_QP_Dense",
        "TE_I": TE_I_l2,
        "TE_O": TE_O_l2,
        "Cons": consistency.consistency(TE_I_l2, TE_O_l2),
        "SupO_EQ": superiority.superiority_oos(TE_O_eq, TE_O_l2),
        "SupO_QP":superiority.superiority_oos(TE_O_qp, TE_O_l2),
        "SupO_L1":superiority.superiority_oos(TE_O_l1, TE_O_l2),
        "SupO_L2":0.0,
        "Sparsity": sparsity.sparsity(w_l2),
        "Runtime": runtime_l2
    })
    # Discrete Genetic Algorithm
    for K in K_values:
        print(f"Running Discrete GA with K={K}")
        start = time.perf_counter()
        w_ga = discrete_ga_sparse.genetic_algo_sparse(
            train_X,
            train_y,
            K=K
        )
        runtime_ga = time.perf_counter() - start
        TE_I_ga, TE_O_ga = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_ga
        )
        results.append({
            "Model": f"GA_K_{K}",
            "TE_I": TE_I_ga,
            "TE_O": TE_O_ga,
            "Cons": consistency.consistency(TE_I_ga, TE_O_ga),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_ga),
            "Sparsity": sparsity.sparsity(w_ga),
            "Runtime": runtime_ga
        })
    # Hybrid Genetic Algorithm - Dense Initialization - Initialize QP Dense
    for K in K_values:
        print(f"Running Discrete Hybrid GA-QP with K={K}")
        start = time.perf_counter()
        w_hgaQP = hybrid_discrete_ga.genetic_algo_sparse_hybrid(
            train_X,train_y,K=K,
            w_dense=w_qp,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
        )
        runtime_hgaQP = time.perf_counter() - start
        TE_I_hgaQP, TE_O_hgaQP = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_hgaQP
        )
        results.append({
            "Model": f"HGA_QP__K_{K}",
            "TE_I": TE_I_hgaQP,
            "TE_O": TE_O_hgaQP,
            "Cons": consistency.consistency(TE_I_hgaQP, TE_O_hgaQP),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_hgaQP),
            "Sparsity": sparsity.sparsity(w_hgaQP),
            "Runtime": runtime_hgaQP
        })
    # Hybrid Genetic Algorithm - Dense Initialization - Initialize L1 QP Soft Sparse
    for K in K_values:
        print(f"Running Discrete Hybrid GA-L1 with K={K}")
        start = time.perf_counter()
        w_hgaL1 = hybrid_discrete_ga.genetic_algo_sparse_hybrid(
            train_X,train_y,K=K,
            w_dense=w_l1,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
        )
        runtime_hgaL1 = time.perf_counter() - start
        TE_I_hgaL1, TE_O_hgaL1 = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_hgaL1
        )
        results.append({
            "Model": f"HGA_L1_K_{K}",
            "TE_I": TE_I_hgaL1,
            "TE_O": TE_O_hgaL1,
            "Cons": consistency.consistency(TE_I_hgaL1, TE_O_hgaL1),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_hgaL1),
            "Sparsity": sparsity.sparsity(w_hgaL1),
            "Runtime": runtime_hgaL1
        })
    # Hybrid Genetic Algorithm - Dense Initialization - Initialize L2 Ridge QP Dense
    for K in K_values:
        print(f"Running Discrete Hybrid GA-L2 with K={K}")
        start = time.perf_counter()
        w_hgaL2 = hybrid_discrete_ga.genetic_algo_sparse_hybrid(
            train_X,train_y,K=K,
            w_dense=w_l2,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
        )
        runtime_hgaL2 = time.perf_counter() - start
        TE_I_hgaL2, TE_O_hgaL2 = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_hgaL2
        )
        results.append({
            "Model": f"HGA_L2_K_{K}",
            "TE_I": TE_I_hgaL2,
            "TE_O": TE_O_hgaL2,
            "Cons": consistency.consistency(TE_I_hgaL2, TE_O_hgaL2),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_hgaL2),
            "Sparsity": sparsity.sparsity(w_hgaL2),
            "Runtime": runtime_hgaL2
        })
    
    # Discrete Particle Swarm Optimization
    for K in K_values:
        print(f"Running Discrete PSO with K={K}")
        start = time.perf_counter()
        w_pso = discrete_pso_sparse.pso_sparse(
            train_X,train_y,K=K,
            swarm_size=swarm_size,
            iterations=iterations
        )
        runtime_pso = time.perf_counter() - start
        TE_I_pso, TE_O_pso = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_pso
        )
        results.append({
            "Model": f"PSO_K_{K}",
            "TE_I": TE_I_pso,
            "TE_O": TE_O_pso,
            "Cons": consistency.consistency(TE_I_pso, TE_O_pso),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_pso),
            "Sparsity": sparsity.sparsity(w_pso),
            "Runtime": runtime_pso
        })
    # Discrete Hybrid Particle Swarm Optimization - Initialize QP Dense
    for K in K_values:
        print(f"Running Discrete Hybrid PSO-QP with K={K}")
        start = time.perf_counter()
        w_hpsoQP = hybrid_discrete_pso.hybrid_pso_sparse(
            train_X,train_y,K=K,
            w_dense=w_qp,
            swarm_size=swarm_size,
            iterations=iterations
        )
        runtime_hpsoQP = time.perf_counter() - start
        TE_I_hpsoQP, TE_O_hpsoQP = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_hpsoQP
        )
        results.append({
            "Model": f"HPSO_QP_K_{K}",
            "TE_I": TE_I_hpsoQP,
            "TE_O": TE_O_hpsoQP,
            "Cons": consistency.consistency(TE_I_hpsoQP, TE_O_hpsoQP),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_hpsoQP),
            "Sparsity": sparsity.sparsity(w_hpsoQP),
            "Runtime": runtime_hpsoQP
        })
    # Discrete Hybrid Particle Swarm Optimization - Initialize L1 QP Soft Sparse
    for K in K_values:
        print(f"Running Discrete Hybrid PSO-L1 with K={K}")
        start = time.perf_counter()
        w_hpsoL1 = hybrid_discrete_pso.hybrid_pso_sparse(
            train_X,train_y,K=K,
            w_dense=w_l1,
            swarm_size=swarm_size,
            iterations=iterations
        )
        runtime_hpsoL1 = time.perf_counter() - start
        TE_I_hpsoL1, TE_O_hpsoL1 = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_hpsoL1
        )
        results.append({
            "Model": f"HPSO_L1_K_{K}",
            "TE_I": TE_I_hpsoL1,
            "TE_O": TE_O_hpsoL1,
            "Cons": consistency.consistency(TE_I_hpsoL1, TE_O_hpsoL1),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_hpsoL1),
            "Sparsity": sparsity.sparsity(w_hpsoL1),
            "Runtime": runtime_hpsoL1
        })
    #  Discrete Hybrid Particle Swarm Optimization - Initialize L2 Ridge QP Dense
    for K in K_values:
        print(f"Running Discrete Hybrid PSO-L2 with K={K}")
        start = time.perf_counter()
        w_hpsoL2 = hybrid_discrete_pso.hybrid_pso_sparse(
            train_X,train_y,K=K,
            w_dense=w_l2,
            swarm_size=swarm_size,
            iterations=iterations
        )
        runtime_hpsoL2 = time.perf_counter() - start
        TE_I_hpsoL2, TE_O_hpsoL2 = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_hpsoL2
        )
        results.append({
            "Model": f"HPSO_L2_K_{K}",
            "TE_I": TE_I_hpsoL2,
            "TE_O": TE_O_hpsoL2,
            "Cons": consistency.consistency(TE_I_hpsoL2, TE_O_hpsoL2),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_hpsoL2),
            "Sparsity": sparsity.sparsity(w_hpsoL2),
            "Runtime": runtime_hpsoL2
        })
    # Hybrid L1/2 Index Tracking
    for K in K_values:
        print(f"Running L1/2 with K={K}")
        start = time.perf_counter()
        w_L12 = l12_sparse.l12_hybrid_index_tracking(
            train_X,train_y,K=K,
        )
        runtime_L12 = time.perf_counter() - start
        TE_I_L12, TE_O_L12 = te_metrics.compute_te_metrics(
            train_X, train_y,
            test_X, test_y,
            w_L12
        )
        results.append({
            "Model": f"L12_K_{K}",
            "TE_I": TE_I_L12,
            "TE_O": TE_O_L12,
            "Cons": consistency.consistency(TE_I_L12, TE_O_L12),
            "SupO": superiority.superiority_oos(TE_O_eq, TE_O_L12),
            "Sparsity": sparsity.sparsity(w_L12),
            "Runtime": runtime_L12
        })

    return pd.DataFrame(results)

for i in range(1, 9):
    print(f"indtrack{i}")
    print(run_or_convex_experiment(f"indtrack{i}", 
                                    K_values=[10, 20, 40, 50],
                                    population_size=50,
                                    generations=50,
                                    mutation_rate=0.1,
                                    swarm_size=30,
                                    iterations=50
                                    ))

# print(f"indtrack8")
# print(run_or_convex_experiment(f"indtrack1", 
#                                     lambda_l1=0.01, 
#                                     lambda_l2=1.0,
#                                     K_values=[5, 10, 20],
#                                     population_size=50,
#                                     generations=50,
#                                     mutation_rate=0.1,
#                                     swarm_size=30,
#                                     iterations=50
#                                     ))