from __future__ import annotations

import argparse
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to hourly CSV (auto or people)")
    ap.add_argument("--output", required=True, help="Path to daily CSV output")
    ap.add_argument("--kind", choices=["auto", "people"], required=True)
    return ap.parse_args()


def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    # accept either "data" or "date"
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
    elif "date" in df.columns:
        df["data"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("No date column found. Expected 'data' or 'date'.")
    if df["data"].isna().any():
        raise ValueError("Some dates could not be parsed. Check input date format.")
    df["data"] = df["data"].dt.date
    return df


def aggregate_people(df: pd.DataFrame) -> pd.DataFrame:
    # sum counts over 24 hours
    sum_cols = [
        "intrare_pers", "iesire_pers", "total_pers",
        "cet_md", "cet_ro", "cet_ua", "cet_ue", "cet_other",
    ]
    for c in sum_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in people data: {c}")

    daily = (
        df.groupby(["pctf", "data"], as_index=False)[sum_cols]
          .sum()
    )
    daily.rename(columns={
        "intrare_pers": "intrare_day",
        "iesire_pers": "iesire_day",
        "total_pers": "total_day",
    }, inplace=True)

    return daily


def aggregate_auto(df: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [
        "intrare_auto", "iesire_auto", "total_auto",
        "car_in", "car_out",
        "bus_in", "bus_out",
        "light_trucks_in", "light_trucks_out",
        "trucks_in", "trucks_out",
    ]
    for c in sum_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in auto data: {c}")

    daily = (
        df.groupby(["pctf", "data"], as_index=False)[sum_cols]
          .sum()
    )
    daily.rename(columns={
        "intrare_auto": "intrare_day",
        "iesire_auto": "iesire_day",
        "total_auto": "total_day",
    }, inplace=True)

    return daily


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(daily["data"])
    daily["zi_sapt"] = d.dt.isocalendar().day.astype(int)      # 1..7
    daily["month"] = d.dt.month.astype(int)                    # 1..12
    daily["is_weekend"] = (daily["zi_sapt"] >= 6).astype(int)  # 0/1
    return daily


def enforce_invariants(daily: pd.DataFrame) -> pd.DataFrame:
    # Always keep totals consistent for daily main targets
    daily["intrare_day"] = daily["intrare_day"].clip(lower=0).astype(int)
    daily["iesire_day"] = daily["iesire_day"].clip(lower=0).astype(int)
    daily["total_day"] = (daily["intrare_day"] + daily["iesire_day"]).astype(int)
    return daily


def main():
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    if "pctf" not in df.columns:
        raise ValueError("Missing column: pctf")

    df = ensure_date(df)

    if args.kind == "people":
        daily = aggregate_people(df)
    else:
        daily = aggregate_auto(df)

    daily = add_calendar_features(daily)
    daily = enforce_invariants(daily)

    # write date as ISO string for stability
    daily["data"] = pd.to_datetime(daily["data"]).dt.strftime("%Y-%m-%d")
    daily = daily.sort_values(["pctf", "data"]).reset_index(drop=True)

    daily.to_csv(args.output, index=False)
    print(f"Saved daily dataset: {args.output} | rows={len(daily)} | pctf={daily['pctf'].nunique()}")


if __name__ == "__main__":
    main()
