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
                             iterations=50):

    stocks_df, index_df = load_OR_index_dataset(index_name)
    print(f"Number of Periods {stocks_df.shape[0]}, Number of Assets {stocks_df.shape[1]}")

    stock_returns, index_returns = convert_prices_to_returns(
        stocks_df,
        index_df
    )

    print("Std Deviation Stock Returns:", stock_returns.std().mean())
    print("Std Deviation Index Returns:", index_returns.std())

    train_X, test_X, train_y, test_y = train_test_split_or(
        stock_returns,
        index_returns
    )

    N_assets = train_X.shape[1]
    K_values = [K for K in K_values if K < N_assets]

    results = []

    # BASELINE MODELS

    # Equal Weight
    w_eq = equal_weight.equal_weight_or(train_X)
    TE_I_eq, TE_O_eq = te_metrics.compute_te_metrics(
        train_X, train_y, test_X, test_y, w_eq
    )

    # # Least Squares
    # w_ls = least_square.least_squares_or(train_X, train_y)
    # TE_I_ls, TE_O_ls = te_metrics.compute_te_metrics(
    #     train_X, train_y, test_X, test_y, w_ls
    # )

    # Dense QP
    w_qp = qp_dense.least_squares_long_only(train_X, train_y)
    TE_I_qp, TE_O_qp = te_metrics.compute_te_metrics(
        train_X, train_y, test_X, test_y, w_qp
    )

    # L1
    w_l1 = l1_qp_soft_sparse.l1_sparse_tracking_or(train_X, train_y)
    TE_I_l1, TE_O_l1 = te_metrics.compute_te_metrics(
        train_X, train_y, test_X, test_y, w_l1
    )

    # L2
    w_l2 = l2_qp_dense.least_squares_ridge_long_only(train_X, train_y)
    TE_I_l2, TE_O_l2 = te_metrics.compute_te_metrics(
        train_X, train_y, test_X, test_y, w_l2
    )

    # Store classical results
    classical_models = [
        ("EqualWeight", w_eq, TE_I_eq, TE_O_eq),
        # ("LeastSquares", w_ls, TE_I_ls, TE_O_ls),
        ("DenseQP", w_qp, TE_I_qp, TE_O_qp),
        ("L1_Soft_Sparse", w_l1, TE_I_l1, TE_O_l1),
        ("L2_QP_Dense", w_l2, TE_I_l2, TE_O_l2)
    ]

    for name, w, TE_I, TE_O in classical_models:
        results.append({
            "Model": name,
            "TE_I": TE_I,
            "TE_O": TE_O,
            "Cons": abs(TE_I - TE_O),
            # "SupO_EQ": superiority.superiority_oos(TE_O_eq, TE_O),
            # "SupO_QP": superiority.superiority_oos(TE_O_qp, TE_O),
            # "SupO_L1": superiority.superiority_oos(TE_O_l1, TE_O),
            # "SupO_L2": superiority.superiority_oos(TE_O_l2, TE_O),
            # "Sparsity": sparsity.sparsity(w),
            # "Runtime": np.nan
        })

    # METAHEURISTICS
    def append_meta_result(model_name, w, runtime):
        TE_I, TE_O = te_metrics.compute_te_metrics(
            train_X, train_y, test_X, test_y, w
        )

        results.append({
            "Model": model_name,
            "TE_I": TE_I,
            "TE_O": TE_O,
            "Cons": abs(TE_I - TE_O),
            # "SupO_EQ": superiority.superiority_oos(TE_O_eq, TE_O),
            # "SupO_QP": superiority.superiority_oos(TE_O_qp, TE_O),
            # "SupO_L1": superiority.superiority_oos(TE_O_l1, TE_O),
            # "SupO_L2": superiority.superiority_oos(TE_O_l2, TE_O),
            # "Sparsity": sparsity.sparsity(w),
            # "Runtime": runtime
        })

    for K in K_values:

        # GA
        print(f"Running Discrete GA with K={K}")
        start = time.perf_counter()
        w_ga = discrete_ga_sparse.genetic_algo_sparse(train_X, train_y, K=K)
        append_meta_result(f"GA_K_{K}", w_ga, time.perf_counter() - start)

        # HGA-QP
        print(f"Running Discrete Hybrid GA-QP with K={K}")
        start = time.perf_counter()
        w_hga_qp = hybrid_discrete_ga.genetic_algo_sparse_hybrid(
            train_X, train_y, K=K, w_dense=w_qp
        )
        append_meta_result(f"HGA_QP_K_{K}", w_hga_qp, time.perf_counter() - start)

        # HGA-L1
        print(f"Running Discrete Hybrid GA-L1 with K={K}")
        start = time.perf_counter()
        w_hga_l1 = hybrid_discrete_ga.genetic_algo_sparse_hybrid(
            train_X, train_y, K=K, w_dense=w_l1
        )
        append_meta_result(f"HGA_L1_K_{K}", w_hga_l1, time.perf_counter() - start)
        
        # HGA-L2
        print(f"Running Discrete Hybrid GA-L2 with K={K}")
        start = time.perf_counter()
        w_hga_l2 = hybrid_discrete_ga.genetic_algo_sparse_hybrid(
            train_X, train_y, K=K, w_dense=w_l2
        )
        append_meta_result(f"HGA_L2_K_{K}", w_hga_l2, time.perf_counter() - start)
        
        # PSO
        print(f"Running Discrete PSO with K={K}")
        start = time.perf_counter()
        w_pso = discrete_pso_sparse.pso_sparse(
            train_X, train_y, K=K,
            swarm_size=swarm_size,
            iterations=iterations
        )
        append_meta_result(f"PSO_K_{K}", w_pso, time.perf_counter() - start)

        # HPSO-QP
        print(f"Running Discrete Hybrid PSO-QP with K={K}")
        start = time.perf_counter()
        w_pso_qp = hybrid_discrete_pso.hybrid_pso_sparse(
            train_X, train_y, K=K,
            swarm_size=swarm_size,
            iterations=iterations,
            w_dense=w_qp
        )
        append_meta_result(f"HPSO_QP_K_{K}", w_pso_qp, time.perf_counter() - start)
        
        # HPSO-L1
        print(f"Running Discrete Hybrid PSO-L1 with K={K}")
        start = time.perf_counter()
        w_pso_l1 = hybrid_discrete_pso.hybrid_pso_sparse(
            train_X, train_y, K=K,
            swarm_size=swarm_size,
            iterations=iterations,
            w_dense=w_l1
        )
        append_meta_result(f"HPSO_L1_K_{K}", w_pso_l1, time.perf_counter() - start)
        
        # HPSO-L2
        print(f"Running Discrete Hybrid PSO-L2 with K={K}")
        start = time.perf_counter()
        w_pso_l2 = hybrid_discrete_pso.hybrid_pso_sparse(
            train_X, train_y, K=K,
            swarm_size=swarm_size,
            iterations=iterations,
            w_dense=w_l2
        )
        append_meta_result(f"HPSO_L2_K_{K}", w_pso_l2, time.perf_counter() - start)

    return pd.DataFrame(results)

for i in range(1, 9):
    print(f"indtrack{i}")
    print(run_or_convex_experiment(f"indtrack{i}", 
                                    K_values=[5, 10, 20, 40, 50],
                                    population_size=50,
                                    generations=50,
                                    mutation_rate=0.1,
                                    swarm_size=30,
                                    iterations=50
                                    ))

# with pd.ExcelWriter("or_results_data.xlsx", engine="openpyxl") as writer:
    
#     for i in range(1, 9):
#         dataset_name = f"indtrack{i}"
        
#         print(f"Running {dataset_name}...")
        
#         df = run_or_convex_experiment(
#             dataset_name,
#             K_values=[5, 10, 20, 40, 50],
#             population_size=50,
#             generations=50,
#             mutation_rate=0.1,
#             swarm_size=30,
#             iterations=50
#         )
        
#         # Round values for clean formatting
#         df = df.round(6)
        
#         df.to_excel(writer, sheet_name=dataset_name, index=False)

# print("All datasets exported to or_results.xlsx")