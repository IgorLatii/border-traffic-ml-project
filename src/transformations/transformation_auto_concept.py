import pandas as pd
import math
import random

# ============================================================
# Synthetic rescaling for aggregated vehicle border traffic
# ------------------------------------------------------------
# This script loads an aggregated dataset (hour x border point)
# and applies a RANDOM anonymization factor (K) in order to
# generate synthetic traffic volumes while preserving
# internal proportional structure.
#
# The resulting dataset is intended for ML/EDA experiments.
# It does not reflect operational values.
# ============================================================

# ========== CONFIGURATION ===================================
INPUT_CSV = "../data/data_auto.csv"          # aggregated input
OUTPUT_CSV = "../data/traffic_auto_hour.csv" # synthetic output
K = random.uniform(1.05, 1.25)                              # example anonymization factor
# NOTE:
#   K may be randomized by users (e.g. uniform range)
#   The chosen value has no interpretation in real-world terms.
# ============================================================

# Load dataset
df = pd.read_csv(INPUT_CSV, low_memory=False)

# Target numeric fields that will be rescaled
NUM_COLS = [
    "intrare_auto", "iesire_auto", "total_auto",
    "car_in", "car_out",
    "bus_in", "bus_out",
    "light_trucks_in", "light_trucks_out",
    "trucks_in", "trucks_out"
]

# Ensure numeric dtype (coerce invalid formats to zero)
for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


# ============================================================
# Row-level proportional anonymization
# ------------------------------------------------------------
# • Scale total volume using K
# • Redistribute inbound/outbound counts proportionally
# • Refit vehicle categories within inbound/outbound totals
#
# All rounding is integer-based; residuals are assigned to
# the final bucket to maintain invariants.
# ============================================================
def rescale_row(row, k):
    total = row["total_auto"]

    # If no traffic — return untouched
    if total == 0:
        return row

    # 1) Generate a synthetic new total
    new_total = int(round(total * k))

    # 2) Split into inbound/outbound
    intrare = row["intrare_auto"]
    iesire = row["iesire_auto"]

    intrare_new = int(math.floor(new_total * intrare / total)) if total > 0 else 0
    iesire_new = new_total - intrare_new

    # 3) Redistribute inbound by category
    in_groups = ["car_in", "bus_in", "light_trucks_in"]
    used_in = 0
    new_in = {}

    for g in in_groups:
        val = row[g]
        # proportional redistribution inside inbound
        new_val = int(math.floor(intrare_new * val / intrare)) if intrare > 0 else 0
        new_in[g] = new_val
        used_in += new_val

    # assign any rounding delta to trucks
    new_in["trucks_in"] = intrare_new - used_in

    # 4) Redistribute outbound by category
    out_groups = ["car_out", "bus_out", "light_trucks_out"]
    used_out = 0
    new_out = {}

    for g in out_groups:
        val = row[g]
        new_val = int(math.floor(iesire_new * val / iesire)) if iesire > 0 else 0
        new_out[g] = new_val
        used_out += new_val

    new_out["trucks_out"] = iesire_new - used_out

    # 5) Apply synthetic values into the row
    row["total_auto"] = new_total
    row["intrare_auto"] = intrare_new
    row["iesire_auto"] = iesire_new

    # update category values
    for k2, v2 in {**new_in, **new_out}.items():
        row[k2] = v2

    return row


# Apply transformation row-by-row
df = df.apply(rescale_row, axis=1, k=K)


# ============================================================
# Explicit invariant fixes
# (just in case of corner cases after rounding)
# ============================================================

# outbound always complements inbound
df["iesire_auto"] = df["total_auto"] - df["intrare_auto"]

# recompute trucks after residual adjustments
df["trucks_in"] = (
    df["intrare_auto"]
    - df["car_in"]
    - df["bus_in"]
    - df["light_trucks_in"]
)

df["trucks_out"] = (
    df["iesire_auto"]
    - df["car_out"]
    - df["bus_out"]
    - df["light_trucks_out"]
)


# ============================================================
# Integrity assertions (structural guarantees)
# ============================================================
assert (df["intrare_auto"] + df["iesire_auto"] == df["total_auto"]).all()
assert (
    df["car_in"] + df["bus_in"] + df["light_trucks_in"] + df["trucks_in"]
    == df["intrare_auto"]
).all()
assert (
    df["car_out"] + df["bus_out"] + df["light_trucks_out"] + df["trucks_out"]
    == df["iesire_auto"]
).all()


# Save synthetic dataset
df.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Synthetic vehicle dataset saved as: {OUTPUT_CSV}")
