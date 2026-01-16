from __future__ import annotations

import argparse
import pandas as pd

from prophet import Prophet

from .utils import (
    parse_hourly_timestamp,
    time_split,
    seasonal_naive_yesterday,
    select_targets,
    mae,
    rmse,
)


def fit_prophet(train_ds_y: pd.DataFrame) -> Prophet:
    """
    train_ds_y columns: ds (datetime), y (float)
    """
    m = Prophet(
        daily_seasonality=True,   # hourly has daily pattern
        weekly_seasonality=True,  # weekly pattern
        yearly_seasonality=True,  # long-range, may help
        changepoint_prior_scale=0.05,
    )
    m.fit(train_ds_y)
    return m


def predict_on_test(m: Prophet, test_ds: pd.DataFrame) -> pd.Series:
    """
    test_ds columns: ds
    returns yhat aligned to test_ds rows
    """
    fc = m.predict(test_ds[["ds"]])
    return fc["yhat"]


def evaluate_prophet_for_target(df: pd.DataFrame, pctf: str, target_col: str, horizon: int, test_ratio: float = 0.2):
    df_pctf = df[df["pctf"] == pctf].sort_values("timestamp").reset_index(drop=True).copy()

    # --- Baseline (yesterday) ---
    y_true_b, y_pred_b = seasonal_naive_yesterday(df_pctf, target_col=target_col, horizon=horizon, season_lag=24)
    base_mae = mae(y_true_b, y_pred_b)
    base_rmse = rmse(y_true_b, y_pred_b)

    # --- Build supervised pairs for prophet evaluation ---
    # We want to predict y(t+h). For each row t, ds = timestamp(t+h), y = target(t+h)
    df_pctf["ds"] = df_pctf["timestamp"].shift(-horizon)
    df_pctf["y"] = df_pctf[target_col].shift(-horizon)

    pairs = df_pctf[["ds", "y"]].dropna().reset_index(drop=True)

    train_pairs, test_pairs = time_split(pairs, test_ratio=test_ratio)

    # Prophet expects y float
    train_pairs = train_pairs.copy()
    train_pairs["y"] = pd.to_numeric(train_pairs["y"], errors="coerce").astype(float)

    test_pairs = test_pairs.copy()
    test_pairs["y"] = pd.to_numeric(test_pairs["y"], errors="coerce").astype(float)

    m = fit_prophet(train_pairs)

    yhat = predict_on_test(m, test_pairs)
    y_true = test_pairs["y"].to_numpy(dtype=float)
    y_pred = yhat.to_numpy(dtype=float)

    rows = [
        {
            "pctf": pctf,
            "target": target_col,
            "horizon": horizon,
            "model": "Baseline_yesterday",
            "mae": base_mae,
            "rmse": base_rmse,
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
        },
        {
            "pctf": pctf,
            "target": target_col,
            "horizon": horizon,
            "model": "Prophet",
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
        },
    ]

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--kind", choices=["auto", "people"], required=True)
    ap.add_argument("--pctf", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df = parse_hourly_timestamp(df)

    targets = select_targets(args.kind)

    all_res = []
    for t in targets:
        all_res.append(evaluate_prophet_for_target(df, args.pctf, t, args.horizon, test_ratio=args.test_ratio))

    out = pd.concat(all_res, ignore_index=True)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
