# =============================================================================
# 01_eda.py
# CX Support Analytics — Phase 1: Data Ingestion, EDA & Cleaning
# Author : Subham Jena
# Created: 2024
# Purpose: Load raw support ticket dataset, perform exploratory analysis,
#          clean and validate data, export a production-ready CSV for the
#          LLM classification pipeline in Phase 2.
# =============================================================================

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # headless backend — no GUI popup
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
RAW_DATA    = BASE_DIR / "data" / "raw"  / "cx_support_datasets.xlsx"
PROCESSED   = BASE_DIR / "data" / "processed"
OUTPUT_EDA  = BASE_DIR / "outputs" / "eda"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUT_EDA.mkdir(parents=True, exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1 — CX Support Analytics: EDA & Cleaning")
print("="*60)

print("\n[1/6] Loading raw dataset...")

sheets = pd.read_excel(RAW_DATA, sheet_name=None)
df          = sheets["fact_tickets"].copy()
df_agents   = sheets["dim_agents"].copy()
df_products = sheets["dim_products"].copy()
df_channels = sheets["dim_channels"].copy()
df_dates    = sheets["dim_date"].copy()

print(f"  fact_tickets : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
print(f"  dim_agents   : {df_agents.shape[0]} rows")
print(f"  dim_products : {df_products.shape[0]} rows")
print(f"  dim_channels : {df_channels.shape[0]} rows")
print(f"  dim_date     : {df_dates.shape[0]} rows")

# ── 2. INSPECT ────────────────────────────────────────────────────────────────
print("\n[2/6] Schema & null inspection...")

print("\n  --- Column types ---")
print(df.dtypes.to_string())

print("\n  --- Null counts ---")
nulls = df.isnull().sum()
print(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "  No nulls found.")

print("\n  --- Duplicate ticket_id count ---")
dupe_count = df.duplicated(subset="ticket_id").sum()
print(f"  {dupe_count} duplicate ticket IDs")

print("\n  --- Numeric summary (resolution_hours) ---")
print(df["resolution_hours"].describe().round(2).to_string())

print("\n  --- Categorical distributions ---")
for col in ["root_cause", "sentiment", "urgency_label", "status", "channel_id", "customer_tier"]:
    print(f"\n  {col}:")
    print(df[col].value_counts().to_string())

# ── 3. CLEAN ──────────────────────────────────────────────────────────────────
print("\n[3/6] Cleaning...")

rows_before = len(df)

# 3a. Drop rows where ticket_text is null — unusable for LLM pipeline
df = df[df["ticket_text"].notna()].copy()

# 3b. Drop exact duplicate ticket IDs — keep first occurrence
df = df.drop_duplicates(subset="ticket_id", keep="first")

# 3c. Parse datetime columns
df["created_at"]  = pd.to_datetime(df["created_at"],  errors="coerce")
df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce")

# 3d. Strip whitespace from all string columns
str_cols = df.select_dtypes(include=["object", "str"]).columns
for col in str_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace("nan", None)

# 3e. Standardise categorical casing
for col in ["root_cause", "sentiment", "status", "customer_tier", "sla_breached", "escalated"]:
    if col in df.columns:
        df[col] = df[col].str.title()

# 3f. Fill nulls in non-critical categoricals
df["channel_id"]   = df["channel_id"].fillna("Unknown")
df["product_id"]   = df["product_id"].fillna("Unknown")
df["customer_tier"]= df["customer_tier"].fillna("Bronze")

# 3g. Recompute resolution_hours from timestamps as validation
mask_resolved = df["resolved_at"].notna() & df["created_at"].notna()
df.loc[mask_resolved, "resolution_hours_check"] = (
    (df.loc[mask_resolved, "resolved_at"] - df.loc[mask_resolved, "created_at"])
    .dt.total_seconds() / 3600
).round(2)

# 3h. Flag anomalies — negative resolution time
negative_res = df[df["resolution_hours_check"] < 0]
if len(negative_res) > 0:
    print(f"  WARNING: {len(negative_res)} tickets with negative resolution time — dropping.")
    df = df[~(df["resolution_hours_check"] < 0)]

rows_after = len(df)
print(f"  Rows before clean : {rows_before:,}")
print(f"  Rows after  clean : {rows_after:,}")
print(f"  Rows removed      : {rows_before - rows_after:,}")

# ── 4. VALIDATE ───────────────────────────────────────────────────────────────
print("\n[4/6] Post-clean validation...")

assert df["ticket_id"].nunique() == len(df), "FAIL: Duplicate ticket IDs remain"
assert df["ticket_text"].notna().all(),       "FAIL: Null ticket_text values remain"
assert df["created_at"].notna().all(),        "FAIL: Null created_at values remain"

fk_agent   = df["agent_id"].isin(df_agents["agent_id"]).mean() * 100
fk_product = df["product_id"].isin(set(df_products["product_id"]) | {"Unknown"}).mean() * 100
fk_channel = df["channel_id"].isin(set(df_channels["channel_id"]) | {"Unknown"}).mean() * 100

print(f"  FK integrity — agent_id   : {fk_agent:.1f}%")
print(f"  FK integrity — product_id : {fk_product:.1f}%")
print(f"  FK integrity — channel_id : {fk_channel:.1f}%")
print("  All assertions passed.")

# ── 5. EDA CHARTS ─────────────────────────────────────────────────────────────
print("\n[5/6] Generating EDA charts...")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})

# Chart 1 — Ticket volume by week
df["week"] = df["created_at"].dt.to_period("W").apply(lambda x: x.start_time)
weekly = df.groupby("week").size().reset_index(name="ticket_count")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(weekly["week"], weekly["ticket_count"], linewidth=1.5, color="#185FA5")
ax.fill_between(weekly["week"], weekly["ticket_count"], alpha=0.15, color="#185FA5")
ax.set_title("Weekly ticket volume (2024)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Week starting"); ax.set_ylabel("Ticket count")
plt.tight_layout()
plt.savefig(OUTPUT_EDA / "01_weekly_volume.png"); plt.close()

# Chart 2 — Root cause distribution
rc_counts = df["root_cause"].value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
rc_counts.plot(kind="barh", ax=ax, color="#185FA5", edgecolor="white")
ax.set_title("Ticket volume by root cause", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Count"); ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_EDA / "02_root_cause.png"); plt.close()

# Chart 3 — Sentiment distribution
sent_counts = df["sentiment"].value_counts()
colors = {"Positive": "#1D9E75", "Neutral": "#888780", "Negative": "#D85A30"}
fig, ax = plt.subplots(figsize=(5, 4))
sent_counts.plot(kind="bar", ax=ax,
                 color=[colors.get(x, "#888780") for x in sent_counts.index],
                 edgecolor="white", rot=0)
ax.set_title("Sentiment distribution", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_EDA / "03_sentiment.png"); plt.close()

# Chart 4 — Resolution hours histogram (resolved tickets only)
resolved = df[df["resolution_hours"].notna() & (df["resolution_hours"] > 0)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(resolved["resolution_hours"].clip(upper=200), bins=40,
        color="#185FA5", edgecolor="white", alpha=0.85)
ax.set_title("Resolution time distribution (hours, capped at 200)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Hours to resolve"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_EDA / "04_resolution_hours.png"); plt.close()

# Chart 5 — SLA breach rate by root cause
sla_df = df[df["sla_breached"].isin(["Yes", "No"])]
sla_rate = (
    sla_df.groupby("root_cause")["sla_breached"]
    .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
    .sort_values(ascending=False)
    .reset_index(name="breach_pct")
)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(sla_rate["root_cause"], sla_rate["breach_pct"],
               color="#D85A30", edgecolor="white")
ax.set_title("SLA breach rate by root cause (%)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Breach rate (%)"); ax.invert_yaxis()
for bar, val in zip(bars, sla_rate["breach_pct"]):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_EDA / "05_sla_breach_by_root_cause.png"); plt.close()

# Chart 6 — Urgency score distribution
urg_counts = df["urgency_label"].value_counts().reindex(
    ["Critical", "High", "Medium", "Low", "Informational"])
urg_colors = ["#D85A30", "#EF9F27", "#185FA5", "#1D9E75", "#888780"]
fig, ax = plt.subplots(figsize=(8, 4))
urg_counts.plot(kind="bar", ax=ax, color=urg_colors, edgecolor="white", rot=0)
ax.set_title("Ticket volume by urgency level",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_EDA / "06_urgency_distribution.png"); plt.close()

print(f"  6 charts saved to outputs/eda/")

# ── 6. EXPORT ─────────────────────────────────────────────────────────────────
print("\n[6/6] Exporting clean dataset...")

export_cols = [
    "ticket_id", "created_at", "resolved_at", "date_id",
    "agent_id", "product_id", "channel_id",
    "ticket_text", "root_cause",
    "urgency_score", "urgency_label", "sentiment",
    "status", "resolution_hours", "sla_target_hrs",
    "sla_breached", "escalated",
    "first_response_hrs", "customer_tier", "reopen_count"
]

df_export = df[export_cols].copy()
out_path  = PROCESSED / "tickets_clean.csv"
df_export.to_csv(out_path, index=False)

print(f"  Exported : {out_path}")
print(f"  Rows     : {len(df_export):,}")
print(f"  Columns  : {len(df_export.columns)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1 COMPLETE")
print("="*60)
print(f"""
  Input  : data/raw/cx_support_datasets.xlsx
  Output : data/processed/tickets_clean.csv
  Charts : outputs/eda/ (6 PNGs)

  Key observations:
  - {df['root_cause'].value_counts().idxmax()} is the highest-volume root cause
  - {df['sentiment'].value_counts().idxmax()} sentiment dominates ticket pool
  - {sla_rate.iloc[0]['root_cause']} has the highest SLA breach rate
    ({sla_rate.iloc[0]['breach_pct']:.1f}%)
  - Median resolution time: {resolved['resolution_hours'].median():.1f} hrs
  - {(df['escalated'] == 'Yes').mean()*100:.1f}% of tickets were escalated

  Ready for Phase 2: LLM classification pipeline.
""")
