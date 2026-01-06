from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from .io import load_hourly_csv
from .features import build_training_frame
from .metrics import mae, rmse


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_model_like_train_one(
    df_pctf: pd.DataFrame,
    target_col: str,
    horizon: int,
    model_name: str = "hgb",
    lags: List[int] | None = None,
):
    """
    Train a model exactly in the same spirit as train_one.py:
    - build lags + calendar features
    - time split 80/20
    - fit model
    Returns: fitted model, feature_cols, quick validation metrics (MAE/RMSE on test split)
    """
    if lags is None:
        lags = [1, 2, 24, 168]

    frame, feature_cols = build_training_frame(df_pctf, target_col=target_col, horizon=horizon, lags=lags)
    if len(frame) < 500:
        raise ValueError(f"Too few rows after feature engineering for pctf={df_pctf['pctf'].iloc[0]} target={target_col}")

    train_df, test_df = time_split(frame, test_ratio=0.2)
    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    if model_name.lower() in ("hgb", "histgradientboosting"):
        model = HistGradientBoostingRegressor(
            random_state=42,
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
        )
    elif model_name.lower() in ("rf", "randomforest"):
        model = RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
        )
    else:
        raise ValueError("model_name must be 'hgb' or 'rf'")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return model, feature_cols, mae(y_test, pred), rmse(y_test, pred)


def predict_day_hourly_sum(
    df_pctf: pd.DataFrame,
    model,
    feature_cols: List[str],
    target_col: str,
    day: datetime,
    horizon: int,
) -> Tuple[int, pd.DataFrame]:
    """
    Predict hourly values for the given day (24 hours) and sum them.
    Important: we do sequential forecasting hour by hour, reusing actual recent history.
    For a demo (single-day), this is acceptable and easy to explain.

    Returns: (total_predicted, detail_df with hour+prediction)
    """
    # Work on a copy to avoid mutating caller DF
    hist = df_pctf.sort_values("timestamp").reset_index(drop=True).copy()

    preds = []
    for h in range(24):
        ts = day + timedelta(hours=h)

        # Ensure we only use history strictly before ts
        hist_before = hist[hist["timestamp"] < ts].copy()
        if len(hist_before) < 200:
            # Not enough history (should not happen for your datasets)
            preds.append(0)
            continue

        # Build features frame from history; take last row as "current state"
        frame, _ = build_training_frame(hist_before, target_col=target_col, horizon=horizon, lags=[1, 2, 24, 168])
        if len(frame) == 0:
            preds.append(0)
            continue

        x_last = frame.iloc[[-1]][feature_cols]
        yhat = float(model.predict(x_last)[0])
        preds.append(max(0, int(round(yhat))))

    detail = pd.DataFrame({
        "hour": list(range(24)),
        "predicted": preds,
    })
    return int(detail["predicted"].sum()), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="e.g. data/traffic_auto_hour.csv")
    ap.add_argument("--pctf", required=True, help="e.g. LEUSENI")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (the day to forecast)")
    ap.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours (e.g., 24 for next-day style)")
    ap.add_argument("--model", choices=["hgb", "rf"], default="hgb", help="Which model to use for demo")
    args = ap.parse_args()

    df = load_hourly_csv(args.csv)
    df_pctf = df[df["pctf"] == args.pctf].sort_values("timestamp").reset_index(drop=True)
    if df_pctf.empty:
        raise ValueError(f"PCTF '{args.pctf}' not found in dataset")

    day = datetime.strptime(args.date, "%Y-%m-%d")

    # Train 2 models: intrare and iesire
    for target_col in ("intrare_auto", "iesire_auto"):
        model, feature_cols, val_mae, val_rmse = train_model_like_train_one(
            df_pctf=df_pctf,
            target_col=target_col,
            horizon=1,               # model is trained for 1-step; we demo day by summing hourly predictions
            model_name=args.model,
        )

        total, detail = predict_day_hourly_sum(
            df_pctf=df_pctf,
            model=model,
            feature_cols=feature_cols,
            target_col=target_col,
            day=day,
            horizon=args.horizon,
        )

        print(f"\n=== {args.pctf} | {target_col} ===")
        print(f"Model: {args.model} | quick validation MAE={val_mae:.2f} RMSE={val_rmse:.2f}")
        print(f"Forecast day: {args.date} | horizon={args.horizon}h")
        print(f"Predicted total for the day (sum of 24 hourly forecasts): {total}")

        # Optional: show top hours
        top = detail.sort_values("predicted", ascending=False).head(23)
        print("\nTop 23 predicted hours:")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
