"""One-step-ahead baselines evaluated on exactly the same target indices as the LSTM."""
from __future__ import annotations

import warnings

import numpy as np


def persistence(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Next hour = this hour."""
    return y[idx - 1]


def seasonal_naive(y: np.ndarray, idx: np.ndarray, period: int = 24) -> np.ndarray:
    """Next hour = same hour yesterday."""
    return y[idx - period]


def sarima_one_step(y: np.ndarray, train_end: int, idx: np.ndarray) -> np.ndarray:
    """SARIMA(1,0,0)(1,0,0,24) fitted on the training slice only, then filtered forward
    with fixed parameters so every prediction uses data up to t-1 and nothing after."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(y[:train_end], order=(1, 0, 0), seasonal_order=(1, 0, 0, 24), trend="c").fit(disp=False)
        full = fit.apply(y)  # same parameters, no refit
    pred = full.get_prediction().predicted_mean  # one-step-ahead predictions for every t
    return np.asarray(pred)[idx]
