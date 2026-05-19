"""
data_loader.py -- Phase 1A: Data Loading & Parsing
===================================================
Loads the three raw CSV datasets, parses data types, extracts derived
columns (Brand_ID), and merges them into a unified analysis-ready DataFrame.

Datasets:
  - touchpoints.csv   (~566K rows) : User journey events
  - user_profiles.csv  (~100K rows) : Persona intelligence
  - campaign_spend.csv (~50 rows)   : Financial layer
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOUCHPOINTS_FILE = DATA_DIR / "touchpoints.csv"
USER_PROFILES_FILE = DATA_DIR / "user_profiles.csv"
CAMPAIGN_SPEND_FILE = DATA_DIR / "campaign_spend.csv"

# Canonical ordering for the conversion funnel
EVENT_ORDER = ["Impression", "Click", "Add-to-Cart", "Purchase"]
EVENT_RANK = {e: i for i, e in enumerate(EVENT_ORDER)}

# Expected columns for validation
TOUCHPOINT_COLS = ["User_ID", "Timestamp", "Campaign_ID", "Channel", "Event_Type"]
PROFILE_COLS = ["User_ID", "Segment", "Trend_Affinity", "Geography"]
SPEND_COLS = [
    "Campaign_ID", "Brand_ID", "Channel",
    "Pricing_Model", "Cost_Rate_INR", "Total_Budget_Allocated",
]


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------
def load_touchpoints(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load and parse the touchpoints journey log.

    Returns
    -------
    pd.DataFrame with columns:
        User_ID (str), Timestamp (datetime64), Campaign_ID (str),
        Channel (category), Event_Type (category), Brand_ID (str),
        Event_Rank (int8)
    """
    fp = Path(filepath) if filepath else TOUCHPOINTS_FILE
    print(f"[DataLoader] Loading touchpoints from {fp.name} ...")

    df = pd.read_csv(
        fp,
        dtype={
            "User_ID": "str",
            "Campaign_ID": "str",
            "Channel": "str",
            "Event_Type": "str",
        },
        parse_dates=["Timestamp"],
        dayfirst=False,
    )

    # Validate columns
    missing = set(TOUCHPOINT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Touchpoints CSV missing columns: {missing}")

    # Derive Brand_ID from User_ID  (U_B01_00000 -> B01)
    df["Brand_ID"] = df["User_ID"].str.extract(r"(B\d{2})", expand=False)

    # Numeric rank for funnel ordering
    df["Event_Rank"] = df["Event_Type"].map(EVENT_RANK).astype("int8")

    # Convert to categoricals for memory efficiency
    for col in ["Channel", "Event_Type", "Brand_ID"]:
        df[col] = df[col].astype("category")

    print(f"[DataLoader]   -> {len(df):,} rows loaded  |  "
          f"{df['User_ID'].nunique():,} unique users  |  "
          f"{df['Brand_ID'].nunique()} brands")
    return df


def load_user_profiles(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load and parse the user profiles / persona intelligence.

    Returns
    -------
    pd.DataFrame with columns:
        User_ID (str), Segment (category), Trend_Affinity (category),
        Geography (category), Brand_ID (str)
    """
    fp = Path(filepath) if filepath else USER_PROFILES_FILE
    print(f"[DataLoader] Loading user profiles from {fp.name} ...")

    df = pd.read_csv(fp, dtype="str")

    missing = set(PROFILE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"User profiles CSV missing columns: {missing}")

    df["Brand_ID"] = df["User_ID"].str.extract(r"(B\d{2})", expand=False)

    for col in ["Segment", "Trend_Affinity", "Geography", "Brand_ID"]:
        df[col] = df[col].astype("category")

    print(f"[DataLoader]   -> {len(df):,} profiles  |  "
          f"Segments: {list(df['Segment'].unique())}  |  "
          f"Geographies: {list(df['Geography'].unique())}")
    return df


def load_campaign_spend(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load and parse the campaign financial layer.

    Returns
    -------
    pd.DataFrame with columns:
        Campaign_ID (str), Brand_ID (str/category), Channel (category),
        Pricing_Model (category), Cost_Rate_INR (float64),
        Total_Budget_Allocated (float64)
    """
    fp = Path(filepath) if filepath else CAMPAIGN_SPEND_FILE
    print(f"[DataLoader] Loading campaign spend from {fp.name} ...")

    df = pd.read_csv(
        fp,
        dtype={
            "Campaign_ID": "str",
            "Brand_ID": "str",
            "Channel": "str",
            "Pricing_Model": "str",
        },
    )

    missing = set(SPEND_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Campaign spend CSV missing columns: {missing}")

    for col in ["Brand_ID", "Channel", "Pricing_Model"]:
        df[col] = df[col].astype("category")

    total_budget = df["Total_Budget_Allocated"].sum()
    print(f"[DataLoader]   -> {len(df)} campaigns  |  "
          f"Total budget: Rs.{total_budget:,.0f}  "
          f"(Rs.{total_budget/1e7:.2f} Cr)")
    return df


# ---------------------------------------------------------------------------
# Merged / unified loader
# ---------------------------------------------------------------------------
def load_all(data_dir: str | Path | None = None):
    """
    Load all three datasets and return them individually + a merged view.

    Returns
    -------
    tuple : (df_touchpoints, df_profiles, df_spend, df_merged)
        df_merged has touchpoints enriched with profile + spend columns.
    """
    if data_dir:
        tp_file = Path(data_dir) / "touchpoints.csv"
        up_file = Path(data_dir) / "user_profiles.csv"
        cs_file = Path(data_dir) / "campaign_spend.csv"
    else:
        tp_file, up_file, cs_file = None, None, None

    df_tp = load_touchpoints(tp_file)
    df_up = load_user_profiles(up_file)
    df_cs = load_campaign_spend(cs_file)

    # --- Merge touchpoints ← profiles (on User_ID) ---
    print("[DataLoader] Merging touchpoints with user profiles ...")
    df_merged = df_tp.merge(
        df_up.drop(columns=["Brand_ID"]),
        on="User_ID",
        how="left",
        validate="many_to_one",
    )

    # --- Merge touchpoints ← spend (on Campaign_ID) ---
    print("[DataLoader] Merging with campaign spend ...")
    spend_cols_to_join = ["Campaign_ID", "Pricing_Model", "Cost_Rate_INR",
                          "Total_Budget_Allocated"]
    df_merged = df_merged.merge(
        df_cs[spend_cols_to_join],
        on="Campaign_ID",
        how="left",
        validate="many_to_one",
    )

    # Quick sanity checks
    null_profiles = df_merged["Segment"].isna().sum()
    null_spend = df_merged["Pricing_Model"].isna().sum()
    if null_profiles:
        print(f"[DataLoader]   WARNING: {null_profiles:,} touchpoints have no matching profile")
    if null_spend:
        print(f"[DataLoader]   WARNING: {null_spend:,} touchpoints have no matching spend record")

    print(f"[DataLoader] OK Merged DataFrame: {len(df_merged):,} rows x "
          f"{len(df_merged.columns)} columns")
    return df_tp, df_up, df_cs, df_merged


# ---------------------------------------------------------------------------
# Data profiling utility
# ---------------------------------------------------------------------------
def profile_data(df_tp, df_up, df_cs):
    """Print a comprehensive profile of the loaded datasets."""
    print("\n" + "=" * 70)
    print("  ROI LENS -- DATA PROFILE REPORT")
    print("=" * 70)

    # --- Touchpoints ---
    print("\n[TOUCHPOINTS]")
    print(f"   Rows            : {len(df_tp):>10,}")
    print(f"   Unique Users    : {df_tp['User_ID'].nunique():>10,}")
    print(f"   Unique Campaigns: {df_tp['Campaign_ID'].nunique():>10}")
    print(f"   Date Range      : {df_tp['Timestamp'].min()} -> {df_tp['Timestamp'].max()}")
    print(f"   Channels        : {sorted(df_tp['Channel'].unique())}")
    print(f"   Event Types     : {sorted(df_tp['Event_Type'].unique())}")
    print(f"   Brands          : {sorted(df_tp['Brand_ID'].unique())}")

    # Event distribution
    print("\n   Event Distribution:")
    ev_counts = df_tp["Event_Type"].value_counts()
    for event in EVENT_ORDER:
        cnt = ev_counts.get(event, 0)
        pct = cnt / len(df_tp) * 100
        print(f"      {event:<15s} : {cnt:>10,}  ({pct:5.1f}%)")

    # --- User Profiles ---
    print("\n[USER PROFILES]")
    print(f"   Rows            : {len(df_up):>10,}")
    print(f"   Segments        : {sorted(df_up['Segment'].unique())}")
    print(f"   Trend Affinities: {sorted(df_up['Trend_Affinity'].unique())}")
    print(f"   Geographies     : {sorted(df_up['Geography'].unique())}")

    # Segment distribution
    print("\n   Segment Distribution:")
    for seg, cnt in df_up["Segment"].value_counts().items():
        print(f"      {seg:<25s} : {cnt:>8,}  ({cnt/len(df_up)*100:5.1f}%)")

    # --- Campaign Spend ---
    print("\n[CAMPAIGN SPEND]")
    print(f"   Campaigns       : {len(df_cs):>10}")
    total = df_cs["Total_Budget_Allocated"].sum()
    print(f"   Total Budget    : Rs.{total:>14,.0f}  (Rs.{total/1e7:.2f} Cr)")
    print(f"   Pricing Models  : {sorted(df_cs['Pricing_Model'].unique())}")

    # Spend by channel
    print("\n   Budget by Channel:")
    ch_spend = df_cs.groupby("Channel")["Total_Budget_Allocated"].sum().sort_values(ascending=False)
    for ch, amt in ch_spend.items():
        print(f"      {ch:<20s} : Rs.{amt:>14,.0f}  ({amt/total*100:5.1f}%)")

    # Spend by brand
    print("\n   Budget by Brand:")
    br_spend = df_cs.groupby("Brand_ID")["Total_Budget_Allocated"].sum().sort_values(ascending=False)
    for br, amt in br_spend.items():
        print(f"      {br:<10s} : Rs.{amt:>14,.0f}  ({amt/total*100:5.1f}%)")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_tp, df_up, df_cs, df_merged = load_all()
    profile_data(df_tp, df_up, df_cs)
    print("\n[DataLoader] Sample merged rows:")
    print(df_merged.head(10).to_string(index=False))
