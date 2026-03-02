import pandas as pd
import numpy as np

def equal_weight_or(stock_returns: pd.DataFrame):
    N = stock_returns.shape[1]
    weights = np.ones(N) / N
    return weights