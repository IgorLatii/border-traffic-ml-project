from __future__ import annotations

import pandas as pd


def load_hourly_csv(path: str) -> pd.DataFrame:
    """
    Loads hourly traffic CSV and creates a unified timestamp column.
    Expected columns: pctf, data, day_of_week, hour, in_count, out_count, total, ...
    """
    df = pd.read_csv(path)

    # Normalize column names to a common schema (in case CSV differs slightly)
    # Required minimal fields:
    required = {"pctf", "data", "hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Parse date; your synthetic CSV likely uses something like "December 1, 2022"
    df["data"] = pd.to_datetime(df["data"], errors="raise")

    df["hour"] = df["hour"].astype(int)

    # Create timestamp = date + hour
    df["timestamp"] = df["data"] + pd.to_timedelta(df["hour"], unit="h")

    # Sort for time-series logic
    df = df.sort_values(["pctf", "timestamp"]).reset_index(drop=True)

    # Sanity check duplicates
    dup = df.duplicated(subset=["pctf", "timestamp"]).sum()
    if dup > 0:
        raise ValueError(f"Found {dup} duplicated rows for pctf+timestamp. Fix data export.")

    return df
