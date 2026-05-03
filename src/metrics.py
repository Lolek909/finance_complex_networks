import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression


def calculate_lead_lag_corr(series_a, series_b, max_lag=5):
    corrs = []
    for lag in range(1, max_lag + 1):
        c = series_a.shift(lag).corr(series_b)
        corrs.append(c)

    if not corrs or np.all(np.isnan(corrs)):
        return 0.0, 0
    max_idx = np.nanargmax(np.abs(corrs))
    best_c = corrs[max_idx]
    return float(best_c) if not np.isnan(best_c) else 0.0, max_idx + 1


def test_volume_price_cross_corr(vol_a, price_b, max_lag=5):
    corrs = [vol_a.shift(lag).corr(price_b) for lag in range(1, max_lag + 1)]
    if not corrs or np.all(np.isnan(corrs)): return 0.0, 0

    max_idx = np.nanargmax(np.abs(corrs))
    best_c = corrs[max_idx]
    return float(best_c) if not np.isnan(best_c) else 0.0, max_idx + 1


def test_granger_causality(target_y, predictor_x, max_lag=5):
    data = pd.concat([target_y, predictor_x], axis=1).dropna()
    if len(data) < 20 or data.iloc[:, 0].std() < 1e-9 or data.iloc[:, 1].std() < 1e-9:
        return 0.0, 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            best_lag, min_p = 0, 1.0

            for lag in range(1, max_lag + 1):
                p_val = gc_res[lag][0]['ssr_chi2test'][1]
                if p_val < min_p:
                    min_p, best_lag = p_val, lag

            score = 1 - min_p if min_p < 0.05 else 0.0
            return score, best_lag
        except Exception:
            return 0.0, 0


def test_mutual_information(series_a, series_b, max_lag=5):
    best_mi, best_lag = 0.0, 0
    for lag in range(1, max_lag + 1):
        data = pd.concat([series_a.shift(lag), series_b], axis=1).dropna()
        if len(data) < 20: continue

        mi = mutual_info_regression(data.iloc[:, 0:1], data.iloc[:, 1], random_state=42)[0]
        if mi > best_mi:
            best_mi, best_lag = mi, lag
    return best_mi, best_lag


def test_volume_price_cross_corr(vol_a, price_b, max_lag=5):
    corrs = [vol_a.shift(lag).corr(price_b) for lag in range(1, max_lag + 1)]
    if not corrs or np.all(np.isnan(corrs)): return 0.0, 0

    max_idx = np.nanargmax(np.abs(corrs))
    best_c = corrs[max_idx]
    return float(best_c) if not np.isnan(best_c) else 0.0, max_idx + 1


def evaluate_pair_unified(price_a, price_b, vol_a=None, max_lag=5):
    corr_weight, corr_lag = calculate_lead_lag_corr(price_a, price_b, max_lag)
    direction = np.sign(corr_weight) if corr_weight != 0 else 1.0

    if vol_a is not None:
        granger_score, g_lag = test_granger_causality(price_b, price_a, max_lag)
        mi_score, mi_lag = test_mutual_information(price_a, price_b, max_lag)
        vol_price_score, vp_lag = test_volume_price_cross_corr(vol_a, price_b, max_lag)

        if granger_score > 0:
            raw_strength = granger_score + mi_score + (abs(vol_price_score) * 0.5)

            normalized_strength = np.tanh(raw_strength)

            final_weight = normalized_strength * direction

            lags = [lag for lag in [g_lag, mi_lag, vp_lag, corr_lag] if lag > 0]
            final_lag = max(set(lags), key=lags.count) if lags else g_lag

            return final_weight, final_lag, granger_score, mi_score, vol_price_score
        return 0.0, 0, 0.0, 0.0, 0.0
    else:
        return corr_weight, corr_lag, 0.0, 0.0, 0.0