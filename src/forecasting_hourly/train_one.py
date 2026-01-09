from __future__ import annotations

import argparse
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .io import load_hourly_csv
from .features import build_training_frame
from .baselines import seasonal_naive_yesterday
from .metrics import mae, rmse


def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def evaluate_models_for_target(df: pd.DataFrame, pctf: str, target_col: str, horizon: int):
    df_pctf = df[df["pctf"] == pctf].sort_values("timestamp").reset_index(drop=True)

    # Baseline
    y_true_b, y_pred_b = seasonal_naive_yesterday(df_pctf, target_col=target_col, horizon=horizon, season_lag=24)
    base_mae = mae(y_true_b, y_pred_b)
    base_rmse = rmse(y_true_b, y_pred_b)

    # ML frame
    lags = [1, 2, 24, 168]  # last 1h, 2h, yesterday, last week
    frame, feature_cols = build_training_frame(df_pctf, target_col=target_col, horizon=horizon, lags=lags)
    train_df, test_df = time_split(frame, test_ratio=0.2)

    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    rows = []

    # Model 1: RandomForest
    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    rows.append({
        "pctf": pctf,
        "target": target_col,
        "horizon": horizon,
        "model": "Baseline_yesterday",
        "mae": base_mae,
        "rmse": base_rmse,
        "n_train": len(train_df),
        "n_test": len(test_df),
    })
    rows.append({
        "pctf": pctf,
        "target": target_col,
        "horizon": horizon,
        "model": "RandomForest",
        "mae": mae(y_test, pred_rf),
        "rmse": rmse(y_test, pred_rf),
        "n_train": len(train_df),
        "n_test": len(test_df),
    })

    # Model 2: HistGradientBoosting (boosting)
    hgb = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
    )
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test)

    rows.append({
        "pctf": pctf,
        "target": target_col,
        "horizon": horizon,
        "model": "HistGradientBoosting",
        "mae": mae(y_test, pred_hgb),
        "rmse": rmse(y_test, pred_hgb),
        "n_train": len(train_df),
        "n_test": len(test_df),
    })

    return pd.DataFrame(rows).sort_values(["target", "model"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pctf", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--kind", choices=["auto", "people"], required=True)
    args = ap.parse_args()

    df = load_hourly_csv(args.csv)

    if args.kind == "auto":
        targets = ["intrare_auto", "iesire_auto"]
    else:
        targets = ["intrare_pers", "iesire_pers"]

    all_results = []
    for t in targets:
        all_results.append(evaluate_models_for_target(df, args.pctf, t, args.horizon))

    out = pd.concat(all_results, ignore_index=True)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
