import pandas as pd
import numpy as np

def least_squares_or(train_X, train_y):
    R = train_X.values
    y = train_y.values
    XtX = R.T @ R
    XtX_inv = np.linalg.pinv(XtX)
    w_ols = XtX_inv @ R.T @ y
    ones = np.ones(len(w_ols))
    adjustment = (
        (ones @ w_ols - 1)
        / (ones @ XtX_inv @ ones)
    )
    w = w_ols - XtX_inv @ ones * adjustment
    return w