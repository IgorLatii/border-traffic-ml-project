from __future__ import annotations
import pandas as pd


def load_daily_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # normalize date column
    if "data" not in df.columns and "date" in df.columns:
        df["data"] = df["date"]

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    if df["data"].isna().any():
        raise ValueError("Some dates could not be parsed in daily CSV.")

    # daily timestamp (date at midnight) for time ordering
    df["timestamp"] = df["data"]

    # ensure int columns
    for c in df.columns:
        if c.endswith("_day") or c in ("zi_sapt", "month", "is_weekend"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df.sort_values(["pctf", "timestamp"]).reset_index(drop=True)
