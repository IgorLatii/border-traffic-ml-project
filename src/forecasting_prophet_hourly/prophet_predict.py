import argparse
import pandas as pd

from prophet import Prophet

from .utils import parse_hourly_timestamp, select_targets


def fit_prophet(df_pctf: pd.DataFrame, target_col: str) -> Prophet:
    train = df_pctf[["timestamp", target_col]].rename(columns={"timestamp": "ds", target_col: "y"}).copy()
    train["y"] = pd.to_numeric(train["y"], errors="coerce").fillna(0).astype(float)

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    m.fit(train)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--kind", choices=["auto", "people"], required=True)
    ap.add_argument("--pctf", required=True)
    ap.add_argument("--target", default=None, help="If empty, predicts both intrare/iesire for that kind")
    ap.add_argument("--hours", type=int, default=24)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df = parse_hourly_timestamp(df)

    df_pctf = df[df["pctf"] == args.pctf].sort_values("timestamp").reset_index(drop=True)
    if df_pctf.empty:
        raise ValueError(f"No rows for pctf={args.pctf}")

    targets = [args.target] if args.target else select_targets(args.kind)

    last_ts = df_pctf["timestamp"].max()

    for t in targets:
        m = fit_prophet(df_pctf, t)

        future = pd.DataFrame({"ds": pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=args.hours, freq="H")})
        fc = m.predict(future)

        fc_out = fc[["ds", "yhat"]].copy()
        fc_out["yhat"] = fc_out["yhat"].round().astype(int)

        total = int(fc_out["yhat"].sum())
        print(f"\n=== Prophet forecast | {args.pctf} | {t} | next {args.hours} hours ===")
        print(f"Start: {future['ds'].iloc[0]}  End: {future['ds'].iloc[-1]}")
        print(f"Predicted total: {total}")
        print(fc_out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
