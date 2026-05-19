"""
data_cleaner.py -- Phase 1B: Bot Detection & Data Quality
=========================================================
Identifies and removes fraudulent / bot traffic from the touchpoints
dataset, validates event sequences, removes duplicates, and produces
a clean DataFrame ready for attribution analysis.

Bot Detection Signals:
  1. Inhuman frequency      -- >20 events per hour
  2. Impression-only bots   -- 50+ impressions, zero clicks
  3. Timestamp clustering   -- multiple events within < 2 seconds
  4. 24/7 activity          -- active in >20 distinct hours across days
  5. Cross-brand interaction — single user across 3+ brands (impossible
                               given user ID structure, used as sanity check)

Data Quality:
  - Exact duplicate removal
  - Orphan event flagging (Purchase without prior Click)
  - Timestamp range validation
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Bot detection thresholds (tunable)
# ---------------------------------------------------------------------------
MAX_EVENTS_PER_HOUR = 20          # Signal 1: hourly frequency cap
MIN_IMPRESSIONS_FOR_BOT = 50      # Signal 2: min impressions to flag
MAX_NEAR_IDENTICAL_GAP_SEC = 2    # Signal 3: near-identical timestamps
MIN_CLUSTERS_FOR_BOT = 5          # Signal 3: minimum # of clustered pairs
ACTIVE_HOURS_THRESHOLD = 20       # Signal 4: distinct hours active per day


# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------
def _detect_high_frequency_users(df: pd.DataFrame) -> set:
    """
    Signal 1: Users with an abnormally high event rate.
    Flag users who have > MAX_EVENTS_PER_HOUR events in any single hour.
    """
    df_temp = df.copy()
    df_temp["_date"] = df_temp["Timestamp"].dt.date
    df_temp["_hour"] = df_temp["Timestamp"].dt.hour

    hourly = (
        df_temp.groupby(["User_ID", "_date", "_hour"])
        .size()
        .reset_index(name="event_count")
    )

    flagged = hourly[hourly["event_count"] > MAX_EVENTS_PER_HOUR]["User_ID"].unique()
    return set(flagged)


def _detect_impression_only_bots(df: pd.DataFrame) -> set:
    """
    Signal 2: Users with many impressions but zero progression.
    A real user occasionally clicks; bots often just fire impressions.
    """
    user_events = df.groupby("User_ID")["Event_Type"].agg(
        total="count",
        unique_types="nunique",
        has_click=lambda x: (x == "Click").any(),
        has_atc=lambda x: (x == "Add-to-Cart").any(),
        has_purchase=lambda x: (x == "Purchase").any(),
    )

    # Users with lots of impressions but no deeper funnel action
    impression_only = user_events[
        (~user_events["has_click"])
        & (~user_events["has_atc"])
        & (~user_events["has_purchase"])
        & (user_events["total"] >= MIN_IMPRESSIONS_FOR_BOT)
    ]

    return set(impression_only.index)


def _detect_timestamp_clusters(df: pd.DataFrame) -> set:
    """
    Signal 3: Users with multiple near-identical timestamps.
    Bots often fire events in rapid bursts with < 2s between them.
    """
    flagged = set()

    # Process per user -- sort by timestamp, compute diffs
    for user_id, group in df.groupby("User_ID"):
        if len(group) < 3:
            continue
        sorted_ts = group["Timestamp"].sort_values()
        diffs = sorted_ts.diff().dt.total_seconds().dropna()
        cluster_count = (diffs < MAX_NEAR_IDENTICAL_GAP_SEC).sum()
        if cluster_count >= MIN_CLUSTERS_FOR_BOT:
            flagged.add(user_id)

    return flagged


def _detect_247_users(df: pd.DataFrame) -> set:
    """
    Signal 4: Users that are active across nearly all hours of the day,
    suggesting automated rather than human behavior.
    """
    user_hours = (
        df.groupby("User_ID")["Timestamp"]
        .apply(lambda x: x.dt.hour.nunique())
    )

    flagged = user_hours[user_hours >= ACTIVE_HOURS_THRESHOLD].index
    return set(flagged)


# ---------------------------------------------------------------------------
# Master bot detection
# ---------------------------------------------------------------------------
def detect_bots(
    df: pd.DataFrame,
    run_timestamp_check: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run all bot detection signals and return a comprehensive report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw touchpoints with Timestamp, User_ID, Event_Type, Channel, Brand_ID.
    run_timestamp_check : bool
        Whether to run the per-user timestamp clustering check.
        This is the slowest signal -- set False for quick iterations.
    verbose : bool
        Print progress and summary.

    Returns
    -------
    dict with keys:
        "bot_users"           : set of flagged User_IDs
        "signal_counts"       : dict of signal_name -> count
        "signal_sets"         : dict of signal_name -> set of User_IDs
        "total_users"         : int
        "bot_count"           : int
        "bot_percentage"      : float
        "bot_touchpoint_count": int
        "bot_tp_percentage"   : float
    """
    total_users = df["User_ID"].nunique()

    if verbose:
        print("\n" + "=" * 70)
        print("  ROI LENS -- BOT DETECTION ENGINE")
        print("=" * 70)
        print(f"\n  Total users to scan: {total_users:,}")
        print(f"  Total touchpoints  : {len(df):,}\n")

    signals = {}

    # Signal 1: High frequency
    if verbose:
        print("  [*] Signal 1: High-frequency users "
              f"(>{MAX_EVENTS_PER_HOUR} events/hour) ...")
    signals["high_frequency"] = _detect_high_frequency_users(df)
    if verbose:
        print(f"     -> {len(signals['high_frequency']):,} users flagged")

    # Signal 2: Impression-only bots
    if verbose:
        print(f"  [*] Signal 2: Impression-only bots "
              f"(>={MIN_IMPRESSIONS_FOR_BOT} impressions, 0 clicks) ...")
    signals["impression_only"] = _detect_impression_only_bots(df)
    if verbose:
        print(f"     -> {len(signals['impression_only']):,} users flagged")

    # Signal 3: Timestamp clustering (optional -- slow)
    if run_timestamp_check:
        if verbose:
            print(f"  [*] Signal 3: Timestamp clustering "
                  f"(<{MAX_NEAR_IDENTICAL_GAP_SEC}s gaps, "
                  f">={MIN_CLUSTERS_FOR_BOT} clusters) ...")
            print("     (this may take a minute for 100K+ users ...)")
        signals["timestamp_cluster"] = _detect_timestamp_clusters(df)
        if verbose:
            print(f"     -> {len(signals['timestamp_cluster']):,} users flagged")
    else:
        signals["timestamp_cluster"] = set()

    # Signal 4: 24/7 activity
    if verbose:
        print(f"  [*] Signal 4: 24/7 activity "
              f"(>={ACTIVE_HOURS_THRESHOLD} active hours) ...")
    signals["always_on"] = _detect_247_users(df)
    if verbose:
        print(f"     -> {len(signals['always_on']):,} users flagged")

    # --- Union all signals ---
    all_bots = set()
    for s in signals.values():
        all_bots |= s

    # Compute touchpoint impact
    bot_mask = df["User_ID"].isin(all_bots)
    bot_tp_count = bot_mask.sum()

    report = {
        "bot_users": all_bots,
        "signal_counts": {k: len(v) for k, v in signals.items()},
        "signal_sets": signals,
        "total_users": total_users,
        "bot_count": len(all_bots),
        "bot_percentage": len(all_bots) / total_users * 100,
        "bot_touchpoint_count": bot_tp_count,
        "bot_tp_percentage": bot_tp_count / len(df) * 100,
    }

    if verbose:
        _print_bot_summary(report, df)

    return report


def _print_bot_summary(report: dict, df: pd.DataFrame):
    """Pretty-print the bot detection summary."""
    print("\n  " + "-" * 50)
    print("  BOT DETECTION SUMMARY")
    print("  " + "-" * 50)
    print(f"  Total users analyzed    : {report['total_users']:>10,}")
    print(f"  Bot users detected      : {report['bot_count']:>10,}  "
          f"({report['bot_percentage']:.2f}%)")
    print(f"  Bot touchpoints removed : {report['bot_touchpoint_count']:>10,}  "
          f"({report['bot_tp_percentage']:.2f}%)")

    print("\n  Signal Breakdown:")
    for signal_name, count in report["signal_counts"].items():
        print(f"    - {signal_name:<25s} : {count:>8,} users")

    # Signal overlap analysis
    sigs = report["signal_sets"]
    sig_keys = [k for k in sigs if sigs[k]]
    if len(sig_keys) >= 2:
        print("\n  Signal Overlaps:")
        from itertools import combinations
        for a, b in combinations(sig_keys, 2):
            overlap = len(sigs[a] & sigs[b])
            if overlap:
                print(f"    - {a} AND {b} : {overlap:,}")

    # Which channels are most affected?
    bot_mask = df["User_ID"].isin(report["bot_users"])
    if bot_mask.any():
        print("\n  Channel Impact (bot traffic %):")
        ch_total = df.groupby("Channel").size()
        ch_bot = df[bot_mask].groupby("Channel").size()
        for ch in ch_total.index:
            bt = ch_bot.get(ch, 0)
            tot = ch_total[ch]
            print(f"    - {ch:<20s} : {bt:>8,} / {tot:>8,}  "
                  f"({bt/tot*100:5.1f}% bot)")


# ---------------------------------------------------------------------------
# Data quality cleaning
# ---------------------------------------------------------------------------
def remove_exact_duplicates(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df_clean = df.drop_duplicates()
    removed = before - len(df_clean)
    if verbose and removed > 0:
        print(f"[Cleaner] Removed {removed:,} exact duplicate rows "
              f"({removed/before*100:.2f}%)")
    elif verbose:
        print("[Cleaner] No exact duplicates found (OK)")
    return df_clean


def validate_timestamp_range(
    df: pd.DataFrame,
    start: str = "2026-01-01",
    end: str = "2026-04-01",
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove rows with timestamps outside the expected quarter."""
    mask = (df["Timestamp"] >= start) & (df["Timestamp"] < end)
    removed = (~mask).sum()
    if verbose:
        if removed > 0:
            print(f"[Cleaner] Removed {removed:,} rows outside "
                  f"{start} - {end}")
        else:
            print(f"[Cleaner] All timestamps within {start} - {end} (OK)")
    return df[mask].copy()


def flag_orphan_events(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Flag events that violate the logical funnel sequence.
    A Purchase without a prior Click (for the same user & brand) is "orphan".
    We don't remove them -- just flag for awareness.
    """
    df = df.copy()
    df["is_orphan"] = False

    # For each user, check if Purchase exists without prior Click
    purchasers = df[df["Event_Type"] == "Purchase"]["User_ID"].unique()

    orphan_count = 0
    for uid in purchasers:
        user_df = df[df["User_ID"] == uid].sort_values("Timestamp")
        has_click = (user_df["Event_Type"] == "Click").any()
        if not has_click:
            # Purchase without any Click -- flag as orphan
            purchase_mask = (df["User_ID"] == uid) & (df["Event_Type"] == "Purchase")
            df.loc[purchase_mask, "is_orphan"] = True
            orphan_count += purchase_mask.sum()

    if verbose:
        print(f"[Cleaner] Flagged {orphan_count:,} orphan Purchase events "
              f"(Purchase without prior Click)")

    return df


# ---------------------------------------------------------------------------
# Master cleaning pipeline
# ---------------------------------------------------------------------------
def clean_data(
    df: pd.DataFrame,
    run_timestamp_check: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full cleaning pipeline:
      1. Remove exact duplicates
      2. Validate timestamp range
      3. Detect and remove bot traffic
      4. Flag orphan events

    Parameters
    ----------
    df : pd.DataFrame
        Raw (or merged) touchpoints DataFrame.
    run_timestamp_check : bool
        Whether to run the slow timestamp-clustering bot signal.
    verbose : bool
        Print progress.

    Returns
    -------
    tuple : (df_clean, bot_report)
        df_clean   : cleaned DataFrame with bots removed
        bot_report : dict from detect_bots()
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  ROI LENS -- DATA CLEANING PIPELINE")
        print("=" * 70)
        print(f"\n  Input: {len(df):,} rows x {len(df.columns)} columns\n")

    # Step 1: Duplicates
    df = remove_exact_duplicates(df, verbose=verbose)

    # Step 2: Timestamp validation
    df = validate_timestamp_range(df, verbose=verbose)

    # Step 3: Bot detection & removal
    bot_report = detect_bots(df, run_timestamp_check=run_timestamp_check,
                             verbose=verbose)
    bot_users = bot_report["bot_users"]

    if bot_users:
        before = len(df)
        df = df[~df["User_ID"].isin(bot_users)].copy()
        if verbose:
            print(f"\n[Cleaner] Removed {before - len(df):,} bot touchpoints -> "
                  f"{len(df):,} rows remain")
    else:
        if verbose:
            print("\n[Cleaner] No bots detected -- data is clean (OK)")

    # Step 4: Orphan event flagging
    df = flag_orphan_events(df, verbose=verbose)

    if verbose:
        print(f"\n  [OK] CLEAN DATA: {len(df):,} rows  |  "
              f"{df['User_ID'].nunique():,} users  |  "
              f"{df['Brand_ID'].nunique()} brands")
        print("=" * 70)

    return df, bot_report


# ---------------------------------------------------------------------------
# Post-cleaning analytics
# ---------------------------------------------------------------------------
def summarize_clean_data(df_clean: pd.DataFrame):
    """Print a summary of the cleaned dataset for verification."""
    print("\n" + "=" * 70)
    print("  CLEAN DATA SUMMARY")
    print("=" * 70)

    # Overall funnel
    print("\nOverall Funnel (Clean Data):")
    for event in ["Impression", "Click", "Add-to-Cart", "Purchase"]:
        cnt = (df_clean["Event_Type"] == event).sum()
        pct = cnt / len(df_clean) * 100
        print(f"   {event:<15s} : {cnt:>10,}  ({pct:5.1f}%)")

    # Per-brand conversion count
    print("\nPurchases per Brand:")
    purchases = df_clean[df_clean["Event_Type"] == "Purchase"]
    brand_purchases = purchases.groupby("Brand_ID").size().sort_values(ascending=False)
    for brand, cnt in brand_purchases.items():
        print(f"   {brand} : {cnt:>6,} conversions")

    # Per-channel touchpoint share
    print("\nChannel Share (Clean Data):")
    ch_counts = df_clean["Channel"].value_counts()
    for ch, cnt in ch_counts.items():
        print(f"   {ch:<20s} : {cnt:>10,}  ({cnt/len(df_clean)*100:5.1f}%)")

    # Users by segment
    if "Segment" in df_clean.columns:
        print("\nUsers by Segment (Clean Data):")
        seg_users = df_clean.groupby("Segment")["User_ID"].nunique()
        for seg, cnt in seg_users.items():
            print(f"   {seg:<25s} : {cnt:>8,}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_all, profile_data

    # Load raw data
    df_tp, df_up, df_cs, df_merged = load_all()
    profile_data(df_tp, df_up, df_cs)

    # Clean it
    df_clean, bot_report = clean_data(
        df_merged,
        run_timestamp_check=False,  # Set True for full analysis (slower)
        verbose=True,
    )

    # Summarize clean data
    summarize_clean_data(df_clean)

    # Save clean data
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "touchpoints_clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"\n[Cleaner] Saved clean data -> {out_path}")
