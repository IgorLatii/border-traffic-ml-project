from __future__ import annotations

"""
This module evaluates Facebook Prophet on DAILY traffic data
using EXACTLY the same train/test time windows as classical ML models
(RandomForest, HistGradientBoosting, baseline).

The goal is to ensure a fair comparison between:
- feature-based ML models (lags, rolling statistics)
- time-series model Prophet (trend + seasonality)
"""

import argparse
import numpy as np
import pandas as pd

from prophet import Prophet

# Project-specific imports
from src.forecasting_daily.io_daily import load_daily_csv
from src.forecasting_daily.features_daily import build_training_frame_daily
from src.forecasting_daily.train_one_daily import evaluate_models_for_target_daily


# ============================================================
# Metrics
# ============================================================
def mae(y_true, y_pred) -> float:
    """
        Mean Absolute Error (MAE).

        Measures the average absolute difference between
        true and predicted values.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    """
        Root Mean Squared Error (RMSE).

        Penalizes larger errors more strongly than MAE.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


# ============================================================
# Utility functions
# ============================================================
def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
        Splits a time-ordered DataFrame into train and test parts.

        IMPORTANT:
        - No shuffling is used.
        - The last `test_ratio` fraction of observations
          is used as the test set.
    """
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def select_targets_daily(kind: str) -> list[str]:
    """
        Selects default target variables for daily datasets.
    """
    if kind in ("auto", "people"):
        return ["intrare_day", "iesire_day"]
    raise ValueError("kind must be 'auto' or 'people'")

# ============================================================
# Prophet model configuration
# ============================================================

def fit_prophet_daily(train_ds_y: pd.DataFrame) -> Prophet:
    """
        Fits a Prophet model on daily data.

        Expected columns:
        - ds : datetime (date)
        - y  : numeric target value

        Model configuration:
        - weekly seasonality enabled
        - yearly seasonality enabled
        - smooth trend changes
    """
    m = Prophet(
        daily_seasonality=False,            # no intra-day patterns (daily data)
        weekly_seasonality=True,            # weekday/weekend effects
        yearly_seasonality=True,            # annual seasonality
        changepoint_prior_scale=0.05,       # conservative trend flexibility
    )
    m.fit(train_ds_y)
    return m

# ============================================================
# Prophet evaluation (FAIR comparison with ML models)
# ============================================================

def evaluate_prophet_for_target_daily(
    df: pd.DataFrame,
    pctf: str,
    target_col: str,
    horizon_days: int,
    test_ratio: float = 0.2,
) -> pd.DataFrame:
    """
    Evaluates Prophet on the SAME prediction horizon and
    the SAME train/test dates as feature-based ML models.

    Key idea:
    ----------
    ML models require lagged and rolling features.
    This removes:
      - early observations (due to lags)
      - last `horizon_days` observations (due to future shift)

    Prophet does NOT require such features.
    However, to ensure a FAIR comparison,
    Prophet is evaluated ONLY on the dates that remain
    after ML feature engineering.
    """
    # --------------------------------------------------------
    # 1. Filter data by border crossing point (PCTF)
    # --------------------------------------------------------
    df_pctf = df[df["pctf"] == pctf].sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # 2. Build supervised ML frame (lags + rolling + horizon)
    # -------------------------
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

    # --------------------------------------------------------
    # 3. Convert ML frame to Prophet format (ds, y)
    # --------------------------------------------------------
    # In the ML frame:
    #   - timestamp = time t
    #   - y = value at (t + horizon_days)
    #
    # Therefore, Prophet's ds must be shifted forward
    # by `horizon_days`.

    pairs = frame[["timestamp", "y"]].copy()
    pairs["ds"] = pairs["timestamp"] + pd.to_timedelta(horizon_days, unit="D")
    pairs = pairs[["ds", "y"]].reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Train/test split (identical to ML models)
    # --------------------------------------------------------
    train_pairs, test_pairs = time_split(pairs, test_ratio=test_ratio)
    if train_pairs.empty or test_pairs.empty:
        raise ValueError(f"Not enough data for Prophet: pctf={pctf}, target={target_col}, horizon_days={horizon_days}")

    # Ensure numeric target
    train_pairs = train_pairs.copy()
    train_pairs["y"] = pd.to_numeric(train_pairs["y"], errors="coerce").astype(float)

    test_pairs = test_pairs.copy()
    test_pairs["y"] = pd.to_numeric(test_pairs["y"], errors="coerce").astype(float)

    # --------------------------------------------------------
    # 5. Fit Prophet and predict on test dates only
    # --------------------------------------------------------
    model = fit_prophet_daily(train_pairs)
    forecast = model.predict(test_pairs[["ds"]])

    y_true = test_pairs["y"].to_numpy(dtype=float)
    y_pred = forecast["yhat"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # 6. Return evaluation results
    # --------------------------------------------------------
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

# ============================================================
# Command-line interface
# ============================================================

def main():
    """
        Entry point for command-line execution.
    """
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

    # Keep identical split with ML models (to train_one_daily.evaluate_models_for_target_daily())
    if abs(args.test_ratio - 0.2) > 1e-9:
        print("[WARN] --test-ratio is forced to 0.2 to match RF/HGB split logic in train_one_daily.py")
        args.test_ratio = 0.2

    # Load dataset
    df = load_daily_csv(args.csv)

    # Determine targets
    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets.strip() else select_targets_daily(args.kind)

    # Run evaluation
    all_results = []
    for t in targets:
        # ML models (baseline + RF + HistGB)
        all_results.append(evaluate_models_for_target_daily(df, args.pctf, t, args.horizon_days))
        # Prophet
        all_results.append(evaluate_prophet_for_target_daily(df, args.pctf, t, args.horizon_days, test_ratio=args.test_ratio))

    # Display results
    out = pd.concat(all_results, ignore_index=True)
    out = out.sort_values(["target", "model"]).reset_index(drop=True)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()