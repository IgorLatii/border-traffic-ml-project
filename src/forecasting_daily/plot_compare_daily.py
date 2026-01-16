from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.forecasting_daily.io_daily import load_daily_csv
from src.forecasting_daily.features_daily import build_training_frame_daily


# ---------------- helpers ----------------
def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def fit_prophet_daily(train_ds_y: pd.DataFrame) -> Prophet:
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    m.fit(train_ds_y)
    return m


def plot_four_panels(
    ds,
    y_true,
    preds: dict[str, np.ndarray],
    title: str,
    out_path: str,
    last_n: int = 120,
):
    ds = pd.to_datetime(ds)
    y_true = np.asarray(y_true, dtype=float)

    if last_n and last_n > 0:
        ds = ds[-last_n:]
        y_true = y_true[-last_n:]
        preds = {k: np.asarray(v)[-last_n:] for k, v in preds.items()}

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    models = [
        ("Baseline_last_week", "Baseline (last week)"),
        ("RandomForest", "Random Forest"),
        ("HistGradientBoosting", "HistGradientBoosting"),
        ("Prophet", "Prophet"),
    ]

    for ax, (key, label) in zip(axes, models):
        ax.plot(ds, y_true, label="Actual", linewidth=2)
        ax.plot(ds, preds[key], label=label, linewidth=2)
        ax.set_ylabel("Traffic")
        ax.legend()
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pctf", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--horizon-days", type=int, default=7)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--plot-last", type=int, default=120)
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    # keep consistent with training scripts
    if abs(args.test_ratio - 0.2) > 1e-9:
        print("[WARN] --test-ratio forced to 0.2")
        args.test_ratio = 0.2

    df = load_daily_csv(args.csv)
    df_pctf = df[df["pctf"] == args.pctf].sort_values("timestamp").reset_index(drop=True)

    # build supervised frame
    lags_days = [1, 7, 14, 30]
    frame, feature_cols = build_training_frame_daily(
        df_pctf,
        target_col=args.target,
        horizon_days=args.horizon_days,
        lags_days=lags_days,
        use_rollings=True,
    )

    if frame.empty:
        raise ValueError("Not enough data after feature engineering")

    train_f, test_f = time_split(frame, test_ratio=args.test_ratio)

    # ground truth
    y_true = test_f["y"].to_numpy(dtype=float)
    ds_plot = (test_f["timestamp"] + pd.to_timedelta(args.horizon_days, unit="D")).to_numpy()

    # ---------- Baseline ----------
    pairs = frame[["timestamp", "y"]].copy()
    pairs["ds"] = pairs["timestamp"] + pd.to_timedelta(args.horizon_days, unit="D")
    pairs = pairs.sort_values("ds").reset_index(drop=True)
    pairs["baseline"] = pairs["y"].shift(7)

    _, test_pairs = time_split(pairs, test_ratio=args.test_ratio)
    yhat_baseline = test_pairs["baseline"].to_numpy(dtype=float)
    yhat_baseline = pd.Series(yhat_baseline).ffill().bfill().to_numpy()

    # ---------- RF ----------
    X_train = train_f[feature_cols].to_numpy(dtype=float)
    y_train = train_f["y"].to_numpy(dtype=float)
    X_test = test_f[feature_cols].to_numpy(dtype=float)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    yhat_rf = rf.predict(X_test)

    # ---------- HGB ----------
    hgb = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=42)
    hgb.fit(X_train, y_train)
    yhat_hgb = hgb.predict(X_test)

    # ---------- Prophet ----------
    train_pairs, test_pairs2 = time_split(pairs[["ds", "y"]].copy(), test_ratio=args.test_ratio)
    train_pairs["y"] = train_pairs["y"].astype(float)
    test_pairs2["y"] = test_pairs2["y"].astype(float)

    m = fit_prophet_daily(train_pairs)
    fc = m.predict(test_pairs2[["ds"]])
    yhat_prophet = fc["yhat"].to_numpy(dtype=float)

    # ---------- plot ----------
    preds = {
        "Baseline_last_week": yhat_baseline,
        "RandomForest": yhat_rf,
        "HistGradientBoosting": yhat_hgb,
        "Prophet": yhat_prophet,
    }

    out_path = os.path.join(
        args.out_dir,
        f"{args.pctf}_{args.target}_h{args.horizon_days}D_4panels.png",
    )

    title = f"{args.pctf} | {args.target} | horizon={args.horizon_days}D"
    plot_four_panels(ds_plot, y_true, preds, title, out_path, last_n=args.plot_last)

    print(f"[PLOT] saved: {out_path}")


if __name__ == "__main__":
    main()
