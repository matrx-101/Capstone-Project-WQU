import pandas as pd
import numpy as np

def te(X, y, w):
    portfolio_returns = X.values @ w
    return np.sqrt(np.mean((portfolio_returns - y.values)**2))