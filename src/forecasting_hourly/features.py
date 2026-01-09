from __future__ import annotations
import math
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["timestamp"].dt.dayofweek  # 0..6
    out["month"] = out["timestamp"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    out["hour_sin"] = out["hour"].apply(lambda h: math.sin(2 * math.pi * h / 24))
    out["hour_cos"] = out["hour"].apply(lambda h: math.cos(2 * math.pi * h / 24))
    return out


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("pctf")[target_col]
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = g.shift(lag)
    return out


def build_training_frame(df_pctf: pd.DataFrame, target_col: str, horizon: int, lags: list[int]):
    out = df_pctf.copy()
    out["y"] = out[target_col].shift(-horizon)

    out = add_time_features(out)
    out = add_lag_features(out, target_col=target_col, lags=lags)

    feature_cols = [
        "hour", "dow", "month", "is_weekend", "hour_sin", "hour_cos",
        *[f"{target_col}_lag_{lag}" for lag in lags],
    ]
    out = out.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    return out, feature_cols
