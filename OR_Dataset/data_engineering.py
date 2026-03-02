import numpy as np
import pandas as pd
from pathlib import Path

def load_OR_index_dataset(index_name: str):
    BASE_DIR = Path("/Users/homebrew/Documents/WQU/Capstone/Dense to Sparse Code")
    DATA_PATH = BASE_DIR / "OR_library" / f"{index_name}.txt"
    with open(DATA_PATH, "r") as f:
        tokens = f.read().split()
    N = int(tokens[0])
    T = int(tokens[1])
    values = np.array(list(map(float, tokens[2:])))
    expected = (N + 1) * (T + 1)
    if len(values) == expected + (T + 1):
        values = values[:expected]

    if len(values) != expected:
        raise ValueError(
            f"Data size mismatch: got {len(values)}, expected {expected}"
        )
    data_matrix = values.reshape((N + 1, T + 1))
    index_prices = data_matrix[0, :]
    stock_prices = data_matrix[1:, :].T
    stock_df = pd.DataFrame(
        stock_prices,
        columns=[f"Stock_{i+1}" for i in range(N)]
    )
    index_df = pd.DataFrame(index_prices, columns=["Index"])
    return stock_df, index_df

def convert_prices_to_returns(stock_df: pd.DataFrame,
                              index_df: pd.DataFrame):

    stock_returns = stock_df.pct_change().dropna()
    index_returns = index_df["Index"].pct_change().dropna()
    return stock_returns, index_returns

def train_test_split_or(stock_returns: pd.DataFrame,
                        index_returns: pd.Series):
    T = len(stock_returns)
    split_point = T // 2
    train_stocks = stock_returns.iloc[:split_point]
    test_stocks  = stock_returns.iloc[split_point:]
    train_index = index_returns.iloc[:split_point]
    test_index  = index_returns.iloc[split_point:]

    return train_stocks, test_stocks, train_index, test_index




# stocks_df, index_df = load_OR_index_dataset("indtrack8")

# stock_returns, index_returns = convert_prices_to_returns(
#     stocks_df,
#     index_df
# )

# print(stock_returns.head())
# print(index_returns.head())

# Compute TE_I on training

# Compute TE_O on testing

# Compute Cons

# Compute SupO