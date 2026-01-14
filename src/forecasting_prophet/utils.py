from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def parse_hourly_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your CSV has:
      - pctf
      - data (string like "December 1, 2022" OR ISO)
      - hour (0..23)
    We convert to timestamp = datetime(date) + hour
    """
    out = df.copy()
    out["data"] = pd.to_datetime(out["data"], errors="coerce")
    if out["data"].isna().any():
        bad = out[out["data"].isna()].head(5)
        raise ValueError(
            "Failed to parse 'data' column to datetime. Sample bad rows:\n"
            + bad.to_string(index=False)
        )
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce").fillna(0).astype(int)
    out["timestamp"] = out["data"] + pd.to_timedelta(out["hour"], unit="h")
    return out


def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def seasonal_naive_yesterday(df_pctf: pd.DataFrame, target_col: str, horizon: int, season_lag: int = 24):
    """
    Baseline for hourly:
      y_hat(t + horizon) = y(t + horizon - 24)
    Returns arrays aligned by position (no pandas index alignment issues).
    """
    df = df_pctf.sort_values("timestamp").reset_index(drop=True).copy()
    y_true = df[target_col].shift(-horizon)
    y_pred = df[target_col].shift(-(horizon - season_lag))
    out = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    return out["y_true"].to_numpy(dtype=float), out["y_pred"].to_numpy(dtype=float)


def select_targets(kind: str) -> list[str]:
    if kind == "auto":
        return ["intrare_auto", "iesire_auto"]
    if kind == "people":
        return ["intrare_pers", "iesire_pers"]
    raise ValueError("kind must be 'auto' or 'people'")
