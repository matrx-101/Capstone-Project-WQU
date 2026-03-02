import pandas as pd
import numpy as np
from OR_TrackingError import tracking_error

def compute_te_metrics(train_X, train_y, test_X, test_y, w):
    TE_I = tracking_error.te(train_X, train_y, w)
    TE_O = tracking_error.te(test_X, test_y, w)
    return round(TE_I, 6), round(TE_O, 6)