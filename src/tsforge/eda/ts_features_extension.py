### TS FEATURES EXTENSION FUNCTIONS FOR ADVANCED TIMESERIES FEATURE ENGINEERING/STATIC EMBEDDING ## #

import antropy
import nolds
import numba
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from statsforecast.models import _intervals

# windows_ops not a PYPI library..
# from window_ops.shift import shift_array


# recreating shift_array function here.
def shift_array(x, lag):
    x = np.asarray(x)
    if lag > 0:
        return np.concatenate([np.full(lag, np.nan), x[:-lag]])
    elif lag < 0:
        return np.concatenate([x[-lag:], np.full(-lag, np.nan)])
    return x.copy()


##### NIXTLA TS FEATURES EXTENSIONS #####
def ADI(x, freq: int):
    values = np.asarray(x, dtype=np.float64).copy()
    intervals = _intervals(values)
    return {"adi": np.nanmean(intervals)}


def permutation_entropy(x, freq: int):
    return {"permutation_entropy": antropy.perm_entropy(x, normalize=True)}


def hurst_exp_dfa(x, freq: int):
    return {"hurst_exp_dfa": nolds.dfa(x)}


def lya_exp(x, freq: int):
    try:
        return {"lya_exp": nolds.lyap_r(x)}
    except:
        return {"lya_exp": np.nan}


@numba.njit(nopython=True)
def _longest_zero_streak_calc(x):
    current_streak = 0
    max_streak = 0

    for i in range(len(x)):
        if x[i] == 0:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0

    return max_streak


def longest_zero_streak(x, freq: int):
    """Calculate the longest streak of consecutive zeros in a time series.

    Args:
        x: Time series values
        freq: Frequency of the time series (not used in this function)

    Returns:
        Dictionary with the longest streak of consecutive zeros
    """
    result = _longest_zero_streak_calc(x)
    return {"longest_zero_streak": result}


def MI_top_k_lags(x, freq):
    max_lag = min(freq, len(x) - 1)
    if max_lag < 1:
        return {"MI_top_k_lags": np.nan}

    target = x[max_lag:]  # lag 0
    lag_matrix = np.column_stack(
        [
            x[max_lag - lag : len(x) - lag]  # shift by `lag`
            for lag in range(1, max_lag + 1)
        ]
    )

    mi_scores = mutual_info_regression(X=lag_matrix, y=target, random_state=42)

    mi_scores = np.sort(mi_scores)[::-1]
    # Return the sum of the top 5 MI scores
    return {"MI_top_k_lags": mi_scores[:5].sum() / mi_scores.sum()}
