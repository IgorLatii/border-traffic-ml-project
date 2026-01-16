from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from prophet import Prophet

from src.forecasting_daily.io_daily import load_daily_csv
from src.forecasting_daily.features_daily import build_training_frame_daily
from src.forecasting_daily.train_one_daily import evaluate_models_for_target_daily


# ---------- Metrics ----------
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


def select_targets_daily(kind: str) -> list[str]:
    if kind in ("auto", "people"):
        return ["intrare_day", "iesire_day"]
    raise ValueError("kind must be 'auto' or 'people'")


def fit_prophet_daily(train_ds_y: pd.DataFrame) -> Prophet:
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    m.fit(train_ds_y)
    return m


def evaluate_prophet_for_target_daily(
    df: pd.DataFrame,
    pctf: str,
    target_col: str,
    horizon_days: int,
    test_ratio: float = 0.2,
) -> pd.DataFrame:
    """
    Evaluate Prophet on EXACT SAME train/test rows used for ML models:
    - build ML supervised frame (lags/rollings/shift)
    - convert it to pairs-space (ds = timestamp + horizon_days)
    - split identically
    - predict only on test ds
    """

    df_pctf = df[df["pctf"] == pctf].sort_values("timestamp").reset_index(drop=True)

    lags_days = [1, 7, 14, 30]
    frame, _feature_cols = build_training_frame_daily(
        df_pctf,
        target_col=target_col,
        horizon_days=horizon_days,
        lags_days=lags_days,
        use_rollings=True,
    )
    if frame.empty:
        raise ValueError(
            f"Not enough data after feature engineering for Prophet: pctf={pctf}, target={target_col}, horizon_days={horizon_days}"
        )

    pairs = frame[["timestamp", "y"]].copy()
    pairs["ds"] = pairs["timestamp"] + pd.to_timedelta(horizon_days, unit="D")
    pairs = pairs[["ds", "y"]].reset_index(drop=True)

    train_pairs, test_pairs = time_split(pairs, test_ratio=test_ratio)
    if train_pairs.empty or test_pairs.empty:
        raise ValueError(f"Not enough data for Prophet: pctf={pctf}, target={target_col}, horizon_days={horizon_days}")

    train_pairs = train_pairs.copy()
    train_pairs["y"] = pd.to_numeric(train_pairs["y"], errors="coerce").astype(float)

    test_pairs = test_pairs.copy()
    test_pairs["y"] = pd.to_numeric(test_pairs["y"], errors="coerce").astype(float)

    m = fit_prophet_daily(train_pairs)
    fc = m.predict(test_pairs[["ds"]])

    y_true = test_pairs["y"].to_numpy(dtype=float)
    y_pred = fc["yhat"].to_numpy(dtype=float)

    return pd.DataFrame(
        [
            {
                "pctf": pctf,
                "target": target_col,
                "horizon_days": horizon_days,
                "model": "Prophet",
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "n_train": len(train_pairs),
                "n_test": len(test_pairs),
            }
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pctf", required=True)
    ap.add_argument("--kind", choices=["auto", "people"], required=True)
    ap.add_argument("--horizon-days", type=int, default=1)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument(
        "--targets",
        default="",
        help="Optional comma-separated list of daily targets (e.g., intrare_day,iesire_day,total_day). If omitted, uses defaults.",
    )
    args = ap.parse_args()

    # Keep identical split to train_one_daily.evaluate_models_for_target_daily()
    if abs(args.test_ratio - 0.2) > 1e-9:
        print("[WARN] --test-ratio is forced to 0.2 to match RF/HGB split logic in train_one_daily.py")
        args.test_ratio = 0.2

    df = load_daily_csv(args.csv)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets.strip() else select_targets_daily(args.kind)

    all_results = []
    for t in targets:
        all_results.append(evaluate_models_for_target_daily(df, args.pctf, t, args.horizon_days))
        all_results.append(evaluate_prophet_for_target_daily(df, args.pctf, t, args.horizon_days, test_ratio=args.test_ratio))

    out = pd.concat(all_results, ignore_index=True)
    out = out.sort_values(["target", "model"]).reset_index(drop=True)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
