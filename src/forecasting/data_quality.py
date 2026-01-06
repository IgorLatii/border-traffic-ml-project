from __future__ import annotations

import pandas as pd


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with missing values count per column.
    """
    missing = df.isna().sum()
    return missing[missing > 0].sort_values(ascending=False)


def check_negative_values(df: pd.DataFrame, count_columns: list[str]) -> dict:
    """
    Checks for negative values in count-based columns.
    """
    result = {}
    for col in count_columns:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                result[col] = int(n_neg)
    return result


def check_hour_range(df: pd.DataFrame) -> bool:
    """
    Checks that hours are between 0 and 23.
    """
    return df["hour"].between(0, 23).all()


def check_complete_hourly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks missing (pctf, date, hour) combinations.
    Returns rows that are missing (should be empty for synthetic data).
    """
    pctf = df["pctf"].unique()
    dates = df["data"].unique()
    hours = range(24)

    full_index = pd.MultiIndex.from_product(
        [pctf, dates, hours],
        names=["pctf", "data", "hour"]
    )

    current_index = df.set_index(["pctf", "data", "hour"]).index
    missing = full_index.difference(current_index)

    return pd.DataFrame(missing.tolist(), columns=["pctf", "data", "hour"])
