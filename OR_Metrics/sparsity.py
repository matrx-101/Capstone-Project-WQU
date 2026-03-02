import pandas as pd
import numpy as np

def sparsity(w, tol=1e-6):
    return np.sum(np.abs(w) > tol)