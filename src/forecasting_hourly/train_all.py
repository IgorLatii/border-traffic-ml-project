from __future__ import annotations

import argparse
import os
import time
from typing import List, Dict, Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from .io import load_hourly_csv
from .features import build_training_frame
from .baselines import seasonal_naive_yesterday
from .metrics import mae, rmse


def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def evaluate_one_pctf_target_horizon(
    df: pd.DataFrame,
    pctf: str,
    target_col: str,
    horizon: int,
    test_ratio: float,
    lags: List[int],
    run_rf: bool,
    run_hgb: bool,
    rf_params: Dict[str, Any],
    hgb_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Returns list of metric rows (baseline + optional RF + optional HGB)
    for a single (pctf, target_col, horizon).
    """
    df_pctf = df[df["pctf"] == pctf].sort_values("timestamp").reset_index(drop=True)

    rows: List[Dict[str, Any]] = []

    # Baseline (yesterday same hour)
    y_true_b, y_pred_b = seasonal_naive_yesterday(df_pctf, target_col=target_col, horizon=horizon, season_lag=24)
    rows.append({
        "pctf": pctf,
        "target": target_col,
        "horizon": horizon,
        "model": "Baseline_yesterday",
        "mae": mae(y_true_b, y_pred_b),
        "rmse": rmse(y_true_b, y_pred_b),
        "n_rows_pctf": len(df_pctf),
        "n_train": None,
        "n_test": None,
    })

    # ML frame (lags + calendar features)
    frame, feature_cols = build_training_frame(df_pctf, target_col=target_col, horizon=horizon, lags=lags)
    if len(frame) < 500:
        # Too little data after shifts/cleaning; still return baseline only
        rows.append({
            "pctf": pctf,
            "target": target_col,
            "horizon": horizon,
            "model": "SKIP_ML_TOO_FEW_ROWS",
            "mae": None,
            "rmse": None,
            "n_rows_pctf": len(df_pctf),
            "n_train": None,
            "n_test": None,
        })
        return rows

    train_df, test_df = time_split(frame, test_ratio=test_ratio)
    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    # RandomForest
    if run_rf:
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        rows.append({
            "pctf": pctf,
            "target": target_col,
            "horizon": horizon,
            "model": "RandomForest",
            "mae": mae(y_test, pred),
            "rmse": rmse(y_test, pred),
            "n_rows_pctf": len(df_pctf),
            "n_train": len(train_df),
            "n_test": len(test_df),
        })

    # HistGradientBoosting (boosting)
    if run_hgb:
        hgb = HistGradientBoostingRegressor(**hgb_params)
        hgb.fit(X_train, y_train)
        pred = hgb.predict(X_test)
        rows.append({
            "pctf": pctf,
            "target": target_col,
            "horizon": horizon,
            "model": "HistGradientBoosting",
            "mae": mae(y_test, pred),
            "rmse": rmse(y_test, pred),
            "n_rows_pctf": len(df_pctf),
            "n_train": len(train_df),
            "n_test": len(test_df),
        })

    return rows


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to hourly CSV")
    ap.add_argument("--kind", choices=["auto", "people"], required=True, help="Dataset type for target selection")
    ap.add_argument("--horizons", default="1,3,24", help="Comma-separated horizons, e.g. 1,3,24")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--out", default="", help="Output CSV path (default: reports/metrics_<kind>.csv)")
    ap.add_argument("--only-hgb", action="store_true", help="Run only baseline + HistGradientBoosting (faster)")
    ap.add_argument("--limit-pctf", type=int, default=0, help="Optional: limit number of PCTFs for quick runs")

    args = ap.parse_args()

    df = load_hourly_csv(args.csv)

    if args.kind == "auto":
        targets = ["intrare_auto", "iesire_auto"]
        default_out = os.path.join("reports", "metrics_auto.csv")
    else:
        targets = ["intrare_pers", "iesire_pers"]
        default_out = os.path.join("reports", "metrics_people.csv")

    horizons = parse_int_list(args.horizons)

    pctf_list = sorted(df["pctf"].unique().tolist())
    if args.limit_pctf and args.limit_pctf > 0:
        pctf_list = pctf_list[: args.limit_pctf]

    os.makedirs("reports", exist_ok=True)

    out_path = args.out.strip() if args.out.strip() else default_out

    # Features config (same as train_one)
    lags = [1, 2, 24, 168]

    run_rf = not args.only_hgb
    run_hgb = True

    rf_params = dict(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )

    hgb_params = dict(
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
    )

    rows_all: List[Dict[str, Any]] = []
    total_tasks = len(pctf_list) * len(targets) * len(horizons)
    done = 0
    t0 = time.time()

    for pctf in pctf_list:
        for target in targets:
            for h in horizons:
                done += 1
                print(f"[{done}/{total_tasks}] pctf={pctf} target={target} horizon={h}")
                rows = evaluate_one_pctf_target_horizon(
                    df=df,
                    pctf=pctf,
                    target_col=target,
                    horizon=h,
                    test_ratio=args.test_ratio,
                    lags=lags,
                    run_rf=run_rf,
                    run_hgb=run_hgb,
                    rf_params=rf_params,
                    hgb_params=hgb_params,
                )
                rows_all.extend(rows)

    result = pd.DataFrame(rows_all)

    # Sort for readability
    result = result.sort_values(["pctf", "target", "horizon", "model"]).reset_index(drop=True)

    result.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\nSaved metrics to: {out_path}")
    print(f"Rows: {len(result)}; elapsed: {elapsed:.1f}s")

    # Optional: quick summary
    # Best model per (pctf,target,horizon) by MAE (ignoring baseline)
    ml_only = result[result["model"].isin(["RandomForest", "HistGradientBoosting"])].copy()
    if len(ml_only) > 0:
        idx = ml_only.groupby(["pctf", "target", "horizon"])["mae"].idxmin()
        best = ml_only.loc[idx].sort_values(["horizon", "target", "mae"])
        best_path = out_path.replace(".csv", "_best.csv")
        best.to_csv(best_path, index=False)
        print(f"Saved best-model summary to: {best_path}")


if __name__ == "__main__":
    main()
