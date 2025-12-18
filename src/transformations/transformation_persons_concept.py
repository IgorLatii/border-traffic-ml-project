import pandas as pd
import math
import random

# ============================================================
# Synthetic anonymization for aggregated hourly border traffic
# ------------------------------------------------------------
# This script loads a (already aggregated) people-traffic dataset
# and applies a RANDOM multiplicative factor K to generate
# synthetic volumes for ML training purposes.
#
# The transformation preserves proportional structure between:
#  - intrare / iesire
#  - key nationality groups
#
# K has no operational meaning and cannot be reversed to
# recover any real-world magnitude.
# ============================================================


# ==================== CONFIG ================================
INPUT_CSV = "../data/data_persoane.csv"
OUTPUT_CSV = "../data/traffic_people_hour.csv"

# Random anonymization factor:
# generates a synthetic global multiplier in [0.9 ; 1.4]
# (users may adjust interval based on ML needs)
K = random.uniform(0.05, 1.25)

# ============================================================


# Load aggregated dataset
df = pd.read_csv(INPUT_CSV, low_memory=False)

# Columns subject to integer rescaling
NUM_COLS = [
    "intrare_pers",
    "iesire_pers",
    "total_pers",
    "cet_md",
    "cet_ro",
    "cet_ua",
    "cet_ue",
    "cet_other"
]

# Ensure dtype consistency
for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


# ============================================================
# Row-level transformation:
# - Apply proportional scaling based on K
# - Redistribute inbound / outbound
# - Redistribute nationality groups
# ============================================================
def rescale_row(row, k):
    total = row["total_pers"]

    # If no traffic, keep structure
    if total == 0 or pd.isna(total):
        return row

    # New anonymized total
    new_total = int(round(total * k))

    # Inbound/Outbound proportional split
    intrare = row["intrare_pers"]
    iesire = row["iesire_pers"]

    if total > 0:
        intrare_new = int(math.floor(new_total * intrare / total))
        iesire_new = new_total - intrare_new
    else:
        intrare_new = iesire_new = 0

    # Nationality bucket redistribution
    groups = ["cet_md", "cet_ro", "cet_ua", "cet_ue"]
    used = 0
    new_values = {}

    for g in groups:
        val = row[g]
        new_val = int(math.floor(new_total * val / total)) if total > 0 else 0
        new_values[g] = new_val
        used += new_val

    # Assign remainder to OTHER
    new_values["cet_other"] = new_total - used

    # Apply synthetic counts back to row
    row["total_pers"] = new_total
    row["intrare_pers"] = intrare_new
    row["iesire_pers"] = iesire_new

    for k2, v2 in new_values.items():
        row[k2] = v2

    return row


# Apply synthetic scaling
df = df.apply(rescale_row, axis=1, k=K)


# ============================================================
# Structural invariants — enforced for safety
# ============================================================
df["intrare_pers"] = df["intrare_pers"].fillna(0).astype(int)
df["iesire_pers"] = df["total_pers"] - df["intrare_pers"]

df["cet_other"] = (
    df["total_pers"]
    - df["cet_md"]
    - df["cet_ro"]
    - df["cet_ua"]
    - df["cet_ue"]
)

# inbound + outbound = total
assert (
    (df["intrare_pers"] + df["iesire_pers"])
    .astype(int)
    .equals(df["total_pers"].astype(int))
)

# nationality buckets sum to total
assert (
    (
        df["cet_md"]
        + df["cet_ro"]
        + df["cet_ua"]
        + df["cet_ue"]
        + df["cet_other"]
    )
    .astype(int)
    .equals(df["total_pers"].astype(int))
)


# Save anonymized dataset
df.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Synthetic dataset saved as: {OUTPUT_CSV}")
