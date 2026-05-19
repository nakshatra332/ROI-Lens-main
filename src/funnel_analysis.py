"""
funnel_analysis.py -- Phase 2: Funnel Analytics & Last-Click Attribution
=========================================================================
Builds conversion funnels per brand x channel, computes the legacy
last-click attribution baseline, and calculates last-click CPA & ROI.

This establishes the "wrong answer" that Nexus currently uses, so we
can later contrast it with the multi-touch attribution truth in Phase 3.

Outputs:
  - Funnel table (Impressions -> Clicks -> ATC -> Purchases per brand x channel)
  - Last-click attribution table (100% credit to last clicked channel)
  - Last-click CPA & ROI per brand x channel
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVENT_ORDER = ["Impression", "Click", "Add-to-Cart", "Purchase"]


# ---------------------------------------------------------------------------
# Phase 2A: Funnel Construction
# ---------------------------------------------------------------------------
def build_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a conversion funnel table per Brand x Channel.

    For each (Brand, Channel) combination, count the number of events
    at each funnel stage and compute stage-to-stage conversion rates.

    Parameters
    ----------
    df : pd.DataFrame
        Clean touchpoints data with columns:
        User_ID, Channel, Event_Type, Brand_ID

    Returns
    -------
    pd.DataFrame with columns:
        Brand_ID, Channel, Impressions, Clicks, CTR(%),
        Add_to_Cart, ATC_Rate(%), Purchases, Conv_Rate(%)
    """
    print("\n" + "=" * 70)
    print("  PHASE 2A: FUNNEL CONSTRUCTION")
    print("=" * 70)

    # Pivot: count events per (Brand, Channel, Event_Type)
    funnel = (
        df.groupby(["Brand_ID", "Channel", "Event_Type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all event columns exist
    for event in EVENT_ORDER:
        if event not in funnel.columns:
            funnel[event] = 0

    # Rename for clarity
    funnel = funnel.rename(columns={
        "Impression": "Impressions",
        "Click": "Clicks",
        "Add-to-Cart": "Add_to_Cart",
        "Purchase": "Purchases",
    })

    # Compute conversion rates (handle division by zero)
    funnel["CTR(%)"] = np.where(
        funnel["Impressions"] > 0,
        funnel["Clicks"] / funnel["Impressions"] * 100,
        0.0
    )
    funnel["ATC_Rate(%)"] = np.where(
        funnel["Clicks"] > 0,
        funnel["Add_to_Cart"] / funnel["Clicks"] * 100,
        0.0
    )
    funnel["Conv_Rate(%)"] = np.where(
        funnel["Impressions"] > 0,
        funnel["Purchases"] / funnel["Impressions"] * 100,
        0.0
    )

    # Reorder columns
    funnel = funnel[[
        "Brand_ID", "Channel",
        "Impressions", "Clicks", "CTR(%)",
        "Add_to_Cart", "ATC_Rate(%)",
        "Purchases", "Conv_Rate(%)",
    ]]

    # Sort for readability
    funnel = funnel.sort_values(["Brand_ID", "Channel"]).reset_index(drop=True)

    # Print summary
    print(f"\n  Funnel table: {len(funnel)} rows (Brand x Channel)")
    print(f"  Total Impressions : {funnel['Impressions'].sum():>10,}")
    print(f"  Total Clicks      : {funnel['Clicks'].sum():>10,}")
    print(f"  Total Add-to-Cart : {funnel['Add_to_Cart'].sum():>10,}")
    print(f"  Total Purchases   : {funnel['Purchases'].sum():>10,}")
    print(f"  Overall CTR       : {funnel['Clicks'].sum()/funnel['Impressions'].sum()*100:.2f}%")
    print(f"  Overall Conv Rate : {funnel['Purchases'].sum()/funnel['Impressions'].sum()*100:.3f}%")

    return funnel


def build_brand_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an aggregated funnel at the brand level (not split by channel).

    Returns
    -------
    pd.DataFrame with columns:
        Brand_ID, Impressions, Clicks, CTR(%), Add_to_Cart,
        ATC_Rate(%), Purchases, Conv_Rate(%)
    """
    funnel = (
        df.groupby(["Brand_ID", "Event_Type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for event in EVENT_ORDER:
        if event not in funnel.columns:
            funnel[event] = 0

    funnel = funnel.rename(columns={
        "Impression": "Impressions",
        "Click": "Clicks",
        "Add-to-Cart": "Add_to_Cart",
        "Purchase": "Purchases",
    })

    funnel["CTR(%)"] = np.where(
        funnel["Impressions"] > 0,
        funnel["Clicks"] / funnel["Impressions"] * 100, 0.0
    )
    funnel["ATC_Rate(%)"] = np.where(
        funnel["Clicks"] > 0,
        funnel["Add_to_Cart"] / funnel["Clicks"] * 100, 0.0
    )
    funnel["Conv_Rate(%)"] = np.where(
        funnel["Impressions"] > 0,
        funnel["Purchases"] / funnel["Impressions"] * 100, 0.0
    )

    funnel = funnel[[
        "Brand_ID", "Impressions", "Clicks", "CTR(%)",
        "Add_to_Cart", "ATC_Rate(%)", "Purchases", "Conv_Rate(%)",
    ]]

    return funnel.sort_values("Brand_ID").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 2B: Last-Click Attribution
# ---------------------------------------------------------------------------
def compute_last_click_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute last-click attribution for every converting user.

    For each user who made a Purchase:
      1. Find the timestamp of their Purchase event.
      2. Look back for the most recent Click before that Purchase.
      3. Attribute 100% credit to the channel of that last Click.

    If a user has a Purchase but no prior Click (orphan), we fall back
    to the channel of the Purchase event itself.

    Parameters
    ----------
    df : pd.DataFrame
        Clean touchpoints with User_ID, Timestamp, Channel, Event_Type, Brand_ID.

    Returns
    -------
    pd.DataFrame with columns:
        User_ID, Brand_ID, Purchase_Timestamp, Last_Click_Channel,
        Last_Click_Timestamp, Credit, Attribution_Type
    """
    print("\n" + "=" * 70)
    print("  PHASE 2B: LAST-CLICK ATTRIBUTION")
    print("=" * 70)

    # Ensure Timestamp is datetime
    if df["Timestamp"].dtype == object:
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Get all purchase events
    purchases = df[df["Event_Type"] == "Purchase"].copy()
    purchases = purchases.sort_values(["User_ID", "Timestamp"])

    # Get all click events
    clicks = df[df["Event_Type"] == "Click"][
        ["User_ID", "Timestamp", "Channel", "Brand_ID"]
    ].copy()
    clicks = clicks.sort_values(["User_ID", "Timestamp"])

    attributions = []
    purchaser_ids = purchases["User_ID"].unique()
    total = len(purchaser_ids)

    print(f"\n  Processing {total:,} converting users ...")

    for i, uid in enumerate(purchaser_ids):
        if (i + 1) % 1000 == 0:
            print(f"    ... {i+1:,} / {total:,} users processed")

        # Get this user's purchases
        user_purchases = purchases[purchases["User_ID"] == uid]
        # Get this user's clicks
        user_clicks = clicks[clicks["User_ID"] == uid]

        for _, purchase_row in user_purchases.iterrows():
            purchase_ts = purchase_row["Timestamp"]
            brand = purchase_row["Brand_ID"]

            # Find clicks BEFORE this purchase
            prior_clicks = user_clicks[user_clicks["Timestamp"] < purchase_ts]

            if len(prior_clicks) > 0:
                # Last click before purchase
                last_click = prior_clicks.iloc[-1]
                attributions.append({
                    "User_ID": uid,
                    "Brand_ID": brand,
                    "Purchase_Timestamp": purchase_ts,
                    "Last_Click_Channel": last_click["Channel"],
                    "Last_Click_Timestamp": last_click["Timestamp"],
                    "Credit": 1.0,
                    "Attribution_Type": "last_click",
                })
            else:
                # Orphan purchase: no prior click -> attribute to purchase channel
                attributions.append({
                    "User_ID": uid,
                    "Brand_ID": brand,
                    "Purchase_Timestamp": purchase_ts,
                    "Last_Click_Channel": purchase_row["Channel"],
                    "Last_Click_Timestamp": pd.NaT,
                    "Credit": 1.0,
                    "Attribution_Type": "orphan_fallback",
                })

    df_attr = pd.DataFrame(attributions)

    # Summary
    total_attr = len(df_attr)
    last_click_count = (df_attr["Attribution_Type"] == "last_click").sum()
    orphan_count = (df_attr["Attribution_Type"] == "orphan_fallback").sum()

    print(f"\n  Total attributions    : {total_attr:,}")
    print(f"  Last-click proper    : {last_click_count:,}  "
          f"({last_click_count/total_attr*100:.1f}%)")
    print(f"  Orphan fallback      : {orphan_count:,}  "
          f"({orphan_count/total_attr*100:.1f}%)")

    return df_attr


def aggregate_last_click(df_attr: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate last-click attributions to get total attributed conversions
    per Brand x Channel.

    Returns
    -------
    pd.DataFrame with columns:
        Brand_ID, Channel, LC_Conversions, LC_Share(%)
    """
    agg = (
        df_attr.groupby(["Brand_ID", "Last_Click_Channel"])["Credit"]
        .sum()
        .reset_index()
        .rename(columns={"Last_Click_Channel": "Channel", "Credit": "LC_Conversions"})
    )

    # Compute per-brand share
    brand_totals = agg.groupby("Brand_ID")["LC_Conversions"].transform("sum")
    agg["LC_Share(%)"] = agg["LC_Conversions"] / brand_totals * 100

    return agg.sort_values(["Brand_ID", "Channel"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 2C: Last-Click CPA & ROI
# ---------------------------------------------------------------------------
def compute_last_click_financials(
    df_attr_agg: pd.DataFrame,
    df_spend: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute CPA and ROI under the last-click attribution model.

    CPA_LC = Channel Spend / Attributed Conversions (last-click)
    ROI_LC = (Attributed Conversions - Spend) / Spend  (simplified, no revenue)

    For a meaningful ROI we'd need revenue-per-purchase, which isn't
    in the dataset. So we report CPA as the primary financial metric.

    Parameters
    ----------
    df_attr_agg : pd.DataFrame
        Aggregated last-click attributions (Brand_ID, Channel, LC_Conversions).
    df_spend : pd.DataFrame
        Campaign spend with Brand_ID, Channel, Total_Budget_Allocated.

    Returns
    -------
    pd.DataFrame with columns:
        Brand_ID, Channel, Budget, LC_Conversions, LC_Share(%),
        CPA_LastClick, Spend_Share(%)
    """
    print("\n" + "=" * 70)
    print("  PHASE 2C: LAST-CLICK FINANCIALS (CPA)")
    print("=" * 70)

    # Aggregate spend by Brand x Channel
    spend_agg = (
        df_spend.groupby(["Brand_ID", "Channel"])["Total_Budget_Allocated"]
        .sum()
        .reset_index()
        .rename(columns={"Total_Budget_Allocated": "Budget"})
    )

    # Merge attribution with spend
    financials = spend_agg.merge(
        df_attr_agg,
        on=["Brand_ID", "Channel"],
        how="left",
    )

    # Fill channels with no attributions
    financials["LC_Conversions"] = financials["LC_Conversions"].fillna(0)
    financials["LC_Share(%)"] = financials["LC_Share(%)"].fillna(0)

    # CPA = Budget / Conversions
    financials["CPA_LastClick"] = np.where(
        financials["LC_Conversions"] > 0,
        financials["Budget"] / financials["LC_Conversions"],
        np.inf,  # No conversions -> infinite CPA
    )

    # Spend share per brand
    brand_budget = financials.groupby("Brand_ID")["Budget"].transform("sum")
    financials["Spend_Share(%)"] = financials["Budget"] / brand_budget * 100

    financials = financials.sort_values(["Brand_ID", "Channel"]).reset_index(drop=True)

    # Print summary
    print("\n  CPA Summary by Brand:")
    for brand in sorted(financials["Brand_ID"].unique()):
        brand_df = financials[financials["Brand_ID"] == brand]
        total_conv = brand_df["LC_Conversions"].sum()
        total_spend = brand_df["Budget"].sum()
        avg_cpa = total_spend / total_conv if total_conv > 0 else float("inf")
        print(f"    {brand}: {total_conv:>6.0f} conversions | "
              f"Budget Rs.{total_spend:>14,.0f} | "
              f"Avg CPA Rs.{avg_cpa:>10,.0f}")

    return financials


# ---------------------------------------------------------------------------
# Phase 2D: Print comprehensive comparison tables
# ---------------------------------------------------------------------------
def print_funnel_report(funnel: pd.DataFrame, brand_funnel: pd.DataFrame):
    """Print a formatted funnel report."""
    print("\n" + "=" * 70)
    print("  FUNNEL REPORT: BRAND-LEVEL OVERVIEW")
    print("=" * 70)

    print(f"\n  {'Brand':<8} {'Impress':>10} {'Clicks':>10} {'CTR%':>8} "
          f"{'ATC':>8} {'Purch':>8} {'Conv%':>8}")
    print("  " + "-" * 62)

    for _, row in brand_funnel.iterrows():
        print(f"  {row['Brand_ID']:<8} {row['Impressions']:>10,} "
              f"{row['Clicks']:>10,} {row['CTR(%)']:>7.2f}% "
              f"{row['Add_to_Cart']:>8,} {row['Purchases']:>8,} "
              f"{row['Conv_Rate(%)']:>7.3f}%")

    # Top and bottom performers
    best = brand_funnel.loc[brand_funnel["Conv_Rate(%)"].idxmax()]
    worst = brand_funnel.loc[brand_funnel["Conv_Rate(%)"].idxmin()]
    print(f"\n  Best converter  : {best['Brand_ID']} "
          f"({best['Conv_Rate(%)']:.3f}%)")
    print(f"  Worst converter : {worst['Brand_ID']} "
          f"({worst['Conv_Rate(%)']:.3f}%)")


def print_attribution_report(
    df_attr_agg: pd.DataFrame,
    financials: pd.DataFrame,
):
    """Print a formatted last-click attribution report."""
    print("\n" + "=" * 70)
    print("  LAST-CLICK ATTRIBUTION REPORT")
    print("=" * 70)

    for brand in sorted(financials["Brand_ID"].unique()):
        brand_fin = financials[financials["Brand_ID"] == brand].copy()
        brand_fin = brand_fin.sort_values("LC_Conversions", ascending=False)

        total_conv = brand_fin["LC_Conversions"].sum()
        total_spend = brand_fin["Budget"].sum()

        print(f"\n  --- {brand} (Total: {total_conv:.0f} purchases, "
              f"Budget Rs.{total_spend:,.0f}) ---")
        print(f"  {'Channel':<20} {'Conv':>8} {'Share%':>8} "
              f"{'Budget':>16} {'CPA':>12}")

        for _, row in brand_fin.iterrows():
            cpa_str = (f"Rs.{row['CPA_LastClick']:>10,.0f}"
                       if row['CPA_LastClick'] != np.inf
                       else "       inf")
            print(f"  {row['Channel']:<20} {row['LC_Conversions']:>8.0f} "
                  f"{row['LC_Share(%)']:>7.1f}% "
                  f"Rs.{row['Budget']:>14,.0f} {cpa_str}")


# ---------------------------------------------------------------------------
# Master Phase 2 runner
# ---------------------------------------------------------------------------
def run_phase2(df_clean: pd.DataFrame, df_spend: pd.DataFrame) -> dict:
    """
    Execute the complete Phase 2 pipeline.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Clean touchpoints from Phase 1.
    df_spend : pd.DataFrame
        Campaign spend data.

    Returns
    -------
    dict with keys:
        "funnel"          : Brand x Channel funnel DataFrame
        "brand_funnel"    : Brand-level funnel DataFrame
        "attributions"    : Per-user last-click attribution DataFrame
        "attr_agg"        : Aggregated last-click per Brand x Channel
        "financials"      : Last-click CPA/ROI DataFrame
    """
    print("\n" + "#" * 70)
    print("  ROI LENS -- PHASE 2: FUNNEL ANALYSIS & LAST-CLICK BASELINE")
    print("#" * 70)

    # 2A: Funnel
    funnel = build_funnel(df_clean)
    brand_funnel = build_brand_funnel(df_clean)
    print_funnel_report(funnel, brand_funnel)

    # 2B: Last-Click Attribution
    df_attr = compute_last_click_attribution(df_clean)
    df_attr_agg = aggregate_last_click(df_attr)

    # 2C: Financials
    financials = compute_last_click_financials(df_attr_agg, df_spend)
    print_attribution_report(df_attr_agg, financials)

    # Summary
    print("\n" + "=" * 70)
    print("  PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"  Funnel rows      : {len(funnel)}")
    print(f"  Attributions     : {len(df_attr):,}")
    print(f"  Financial rows   : {len(financials)}")

    return {
        "funnel": funnel,
        "brand_funnel": brand_funnel,
        "attributions": df_attr,
        "attr_agg": df_attr_agg,
        "financials": financials,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_all
    from src.data_cleaner import clean_data

    # Load and clean
    df_tp, df_up, df_cs, df_merged = load_all()
    df_clean, bot_report = clean_data(
        df_merged, run_timestamp_check=False, verbose=True
    )

    # Run Phase 2
    results = run_phase2(df_clean, df_cs)

    # Save outputs
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results["funnel"].to_csv(out_dir / "funnel_brand_channel.csv", index=False)
    results["brand_funnel"].to_csv(out_dir / "funnel_brand.csv", index=False)
    results["attributions"].to_csv(out_dir / "last_click_attributions.csv", index=False)
    results["attr_agg"].to_csv(out_dir / "last_click_aggregated.csv", index=False)
    results["financials"].to_csv(out_dir / "last_click_financials.csv", index=False)

    print(f"\n[Phase 2] All outputs saved to {out_dir}")
