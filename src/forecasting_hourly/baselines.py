from __future__ import annotations
import pandas as pd


def seasonal_naive_yesterday(df_pctf: pd.DataFrame, target_col: str, horizon: int = 1, season_lag: int = 24):
    """
    Predict y(t+h) using y(t+h-season_lag).
    Returns aligned arrays (y_true, y_pred) for metric computation.
    """
    y_true = df_pctf[target_col].shift(-horizon)
    # shift(season_lag - horizon) gives value at (t+h-season_lag)
    y_pred = df_pctf[target_col].shift(season_lag - horizon)
    return y_true, y_pred
