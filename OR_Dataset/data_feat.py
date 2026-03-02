import numpy as np
import pandas as pd
from numpy.linalg import svd, matrix_rank, cond
from OR_Dataset.data_engineering import (
    load_OR_index_dataset,
    convert_prices_to_returns,
    train_test_split_or
)

# Core Diagnostic Function
def analyze_or_dataset(index_name):
    stocks_df, index_df = load_OR_index_dataset(index_name)
    stock_returns, index_returns = convert_prices_to_returns(
        stocks_df, index_df
    )
    train_X, test_X, train_y, test_y = train_test_split_or(
        stock_returns, index_returns
    )

    R = train_X.values
    T, N = R.shape

    # SVD
    s = svd(R, compute_uv=False)

    smallest_sv = s.min()
    largest_sv = s.max()
    condition_number = largest_sv / smallest_sv

    # Covariance conditioning
    cov = R.T @ R
    cov_s = svd(cov, compute_uv=False)
    cov_condition = cov_s.max() / cov_s.min()

    # Rank
    rank_R = matrix_rank(R)

    # Correlation structure
    corr_matrix = train_X.corr().values
    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    avg_abs_corr = np.mean(np.abs(upper_tri))


    # Effective rank (optional but powerful)
    normalized_s = s / s.sum()
    entropy = -np.sum(normalized_s * np.log(normalized_s + 1e-12))
    effective_rank = np.exp(entropy)

    # Store results
    results = {
        "Dataset": index_name,
        "T_periods": T,
        "N_assets": N,
        "Rank_R": rank_R,
        "Min(T,N)": min(T, N),
        "Smallest_SV": smallest_sv,
        "Largest_SV": largest_sv,
        "Condition_Number_R": condition_number,
        "Condition_Number_RtR": cov_condition,
        "Avg_Abs_Correlation": avg_abs_corr,
        "Effective_Rank": effective_rank
    }

    return results

all_results = []

for i in range(1, 9):
    dataset_name = f"indtrack{i}"
    print(f"Analyzing {dataset_name}")
    result = analyze_or_dataset(dataset_name)
    all_results.append(result)

diagnostics_df = pd.DataFrame(all_results)
print(diagnostics_df)

with pd.ExcelWriter("or_dataset_diagnostics.xlsx", engine="openpyxl") as writer:
    diagnostics_df.round(6).to_excel(writer, sheet_name="Diagnostics", index=False)