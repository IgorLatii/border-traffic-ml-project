from __future__ import annotations
import pandas as pd


def build_training_frame_daily(
    df_pctf: pd.DataFrame,
    target_col: str,
    horizon_days: int,
    lags_days: list[int] | None = None,
    use_rollings: bool = True,
):
    """
    Builds a supervised learning frame from a daily time series.

    y(t) = target(t + horizon_days)
    X(t) includes:
      - lagged targets: target(t - lag)
      - calendar: zi_sapt, month, is_weekend
      - optional rolling means of target
    """
    if lags_days is None:
        lags_days = [1, 7, 14, 30]  # yesterday, last week, 2 weeks, last month

    df = df_pctf.sort_values("timestamp").reset_index(drop=True).copy()

    # target to predict (future)
    df["y"] = df[target_col].shift(-horizon_days)

    # lag features (past)
    for lag in lags_days:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    feature_cols = [f"lag_{lag}" for lag in lags_days]

    # calendar features
    for c in ["zi_sapt", "month", "is_weekend"]:
        if c in df.columns:
            feature_cols.append(c)

    # rolling features
    if use_rollings:
        # Rolling means capture local trend on daily series
        df["roll_7_mean"] = df[target_col].shift(1).rolling(7).mean()
        df["roll_14_mean"] = df[target_col].shift(1).rolling(14).mean()
        feature_cols += ["roll_7_mean", "roll_14_mean"]

    # drop rows where any feature or y is NaN
    frame = df.dropna(subset=feature_cols + ["y"]).copy()

    return frame, feature_cols
