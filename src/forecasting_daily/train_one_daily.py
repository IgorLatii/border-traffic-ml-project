from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .io_daily import load_daily_csv
from .features_daily import build_training_frame_daily


# ---------- Metrics (positional, no pandas index alignment) ----------
def mae(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


# ---------- Helpers ----------
def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def seasonal_naive_last_week(df_pctf: pd.DataFrame, target_col: str, horizon_days: int, season_lag: int = 7):
    """
    Baseline: y_hat(t + horizon) = y(t + horizon - season_lag)
    For daily, season_lag=7 means "same weekday last week".
    """
    df = df_pctf.sort_values("timestamp").reset_index(drop=True).copy()
    y_true = df[target_col].shift(-horizon_days)
    y_pred = df[target_col].shift(-(horizon_days - season_lag))
    out = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    return out["y_true"].to_numpy(dtype=float), out["y_pred"].to_numpy(dtype=float)


def _debug_nonfinite(name: str, arr: np.ndarray):
    nan = np.isnan(arr).sum()
    inf = np.isinf(arr).sum()
    mn = np.nanmin(arr) if arr.size else np.nan
    mx = np.nanmax(arr) if arr.size else np.nan
    print(f"{name}: shape={arr.shape} nan={nan} inf={inf} min={mn:.3f} max={mx:.3f}")


def evaluate_models_for_target_daily(df: pd.DataFrame, pctf: str, target_col: str, horizon_days: int, debug: bool = False):
    df_pctf = df[df["pctf"] == pctf].sort_values("timestamp").reset_index(drop=True)

    # Baseline
    y_true_b, y_pred_b = seasonal_naive_last_week(df_pctf, target_col, horizon_days, season_lag=7)
    base_mae = mae(y_true_b, y_pred_b)
    base_rmse = rmse(y_true_b, y_pred_b)

    # ML frame
    lags_days = [1, 7, 14, 30]
    frame, feature_cols = build_training_frame_daily(
        df_pctf,
        target_col=target_col,
        horizon_days=horizon_days,
        lags_days=lags_days,
        use_rollings=True,
    )

    # Ensure numeric (defensive)
    for c in feature_cols + ["y"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")

    frame = frame.dropna(subset=feature_cols + ["y"]).copy()

    train_df, test_df = time_split(frame, test_ratio=0.2)
    if train_df.empty or test_df.empty:
        raise ValueError(f"Not enough data for pctf={pctf}, target={target_col}, horizon_days={horizon_days}")

    # IMPORTANT: use numpy arrays to avoid index alignment issues everywhere
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["y"].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["y"].to_numpy(dtype=float)

    if debug:
        print("\n[DEBUG] feature_cols:", feature_cols)
        _debug_nonfinite("y_train", y_train)
        _debug_nonfinite("y_test", y_test)
        _debug_nonfinite("X_train", X_train)
        _debug_nonfinite("X_test", X_test)

    rows = []

    rows.append({
        "pctf": pctf,
        "target": target_col,
        "horizon_days": horizon_days,
        "model": "Baseline_last_week",
        "mae": base_mae,
        "rmse": base_rmse,
        "n_train": len(train_df),
        "n_test": len(test_df),
    })

    # RandomForest
    rf = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test).astype(float)

    if debug:
        _debug_nonfinite("pred_rf", pred_rf)

    rows.append({
        "pctf": pctf,
        "target": target_col,
        "horizon_days": horizon_days,
        "model": "RandomForest",
        "mae": mae(y_test, pred_rf),
        "rmse": rmse(y_test, pred_rf),
        "n_train": len(train_df),
        "n_test": len(test_df),
    })

    # HistGradientBoosting
    hgb = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
    )
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test).astype(float)

    if debug:
        _debug_nonfinite("pred_hgb", pred_hgb)

    rows.append({
        "pctf": pctf,
        "target": target_col,
        "horizon_days": horizon_days,
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
    ap.add_argument("--horizon-days", type=int, default=1)
    ap.add_argument("--kind", choices=["auto", "people"], required=True)
    ap.add_argument("--debug", action="store_true", help="Print debug info about arrays/features")
    args = ap.parse_args()

    df = load_daily_csv(args.csv)

    targets = ["intrare_day", "iesire_day"]

    all_results = []
    for t in targets:
        all_results.append(evaluate_models_for_target_daily(df, args.pctf, t, args.horizon_days, debug=args.debug))

    out = pd.concat(all_results, ignore_index=True)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
