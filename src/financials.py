"""
financials.py -- Phase 4: Financial Layer (True CPA & Ad Fatigue)
==================================================================
Computes the true unit economics using attribution-weighted conversions,
contrasts them with the flawed last-click CPA, and detects ad fatigue
via saturation curve fitting.

Outputs:
  - Actual cost calculations (CPC vs CPM verification)
  - True CPA using Markov and Shapley attribution weights
  - CPA comparison table (Last-Click vs Markov vs Shapley)
  - Ad fatigue / saturation curves per brand x channel
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Phase 4A: Cost Verification
# ---------------------------------------------------------------------------
def verify_costs(
    df_clean: pd.DataFrame,
    df_spend: pd.DataFrame,
) -> pd.DataFrame:
    """
    Verify actual costs per campaign by computing expected cost from
    event counts and pricing model, then comparing to allocated budget.

    For CPC campaigns: Expected Cost = Clicks x Cost_Rate_INR
    For CPM campaigns: Expected Cost = (Impressions / 1000) x Cost_Rate_INR

    Parameters
    ----------
    df_clean : pd.DataFrame
        Clean touchpoints.
    df_spend : pd.DataFrame
        Campaign spend data.

    Returns
    -------
    pd.DataFrame with cost verification per campaign.
    """
    print("\n" + "=" * 70)
    print("  PHASE 4A: COST VERIFICATION")
    print("=" * 70)

    # Count impressions and clicks per campaign
    impressions = (
        df_clean[df_clean["Event_Type"] == "Impression"]
        .groupby("Campaign_ID")
        .size()
        .reset_index(name="Impression_Count")
    )
    clicks = (
        df_clean[df_clean["Event_Type"] == "Click"]
        .groupby("Campaign_ID")
        .size()
        .reset_index(name="Click_Count")
    )

    # Merge with spend
    cost_df = df_spend.copy()
    cost_df = cost_df.merge(impressions, on="Campaign_ID", how="left")
    cost_df = cost_df.merge(clicks, on="Campaign_ID", how="left")
    cost_df["Impression_Count"] = cost_df["Impression_Count"].fillna(0).astype(int)
    cost_df["Click_Count"] = cost_df["Click_Count"].fillna(0).astype(int)

    # Compute expected cost based on pricing model
    cost_df["Expected_Cost"] = np.where(
        cost_df["Pricing_Model"] == "CPC",
        cost_df["Click_Count"] * cost_df["Cost_Rate_INR"],
        (cost_df["Impression_Count"] / 1000) * cost_df["Cost_Rate_INR"],
    )

    # Budget utilization
    cost_df["Budget_Util(%)"] = (
        cost_df["Expected_Cost"] / cost_df["Total_Budget_Allocated"] * 100
    )

    # Print summary
    print(f"\n  {'Campaign':<20s} {'Model':>5} {'Budget':>16} "
          f"{'Expected':>16} {'Util%':>8}")
    print("  " + "-" * 70)
    for _, row in cost_df.iterrows():
        print(f"  {row['Campaign_ID']:<20s} {row['Pricing_Model']:>5} "
              f"Rs.{row['Total_Budget_Allocated']:>13,.0f} "
              f"Rs.{row['Expected_Cost']:>13,.0f} "
              f"{row['Budget_Util(%)']:>7.1f}%")

    avg_util = cost_df["Budget_Util(%)"].mean()
    print(f"\n  Average budget utilization: {avg_util:.1f}%")

    return cost_df


# ---------------------------------------------------------------------------
# Phase 4B: True CPA (Multi-Touch)
# ---------------------------------------------------------------------------
def compute_true_cpa(
    df_spend: pd.DataFrame,
    df_lc_financials: pd.DataFrame,
    df_markov: pd.DataFrame,
    df_shapley: pd.DataFrame,
    total_conversions_by_brand: pd.Series,
) -> pd.DataFrame:
    """
    Compute CPA under all three attribution models.

    True CPA = Channel Spend / (Total Brand Conversions x Attribution %)

    Parameters
    ----------
    df_spend : pd.DataFrame
        Campaign spend.
    df_lc_financials : pd.DataFrame
        Last-click financials from Phase 2.
    df_markov : pd.DataFrame
        Markov attribution results.
    df_shapley : pd.DataFrame
        Shapley attribution results.
    total_conversions_by_brand : pd.Series
        Total conversions per brand.

    Returns
    -------
    pd.DataFrame with CPA under all three models.
    """
    print("\n" + "=" * 70)
    print("  PHASE 4B: TRUE CPA (MULTI-TOUCH)")
    print("=" * 70)

    # Aggregate spend by Brand x Channel
    spend_agg = (
        df_spend.groupby(["Brand_ID", "Channel"])["Total_Budget_Allocated"]
        .sum()
        .reset_index()
        .rename(columns={"Total_Budget_Allocated": "Budget"})
    )

    # Merge Markov attribution
    markov_cols = df_markov[["Brand_ID", "Channel", "Markov_Attribution(%)"]].copy()

    # Merge Shapley attribution
    shapley_cols = df_shapley[["Brand_ID", "Channel", "Shapley_Attribution(%)"]].copy()

    # Last-click from financials
    lc_cols = df_lc_financials[["Brand_ID", "Channel", "LC_Conversions",
                                 "CPA_LastClick"]].copy()

    # Build the comparison table
    cpa_df = spend_agg.copy()
    cpa_df = cpa_df.merge(lc_cols, on=["Brand_ID", "Channel"], how="left")
    cpa_df = cpa_df.merge(markov_cols, on=["Brand_ID", "Channel"], how="left")
    cpa_df = cpa_df.merge(shapley_cols, on=["Brand_ID", "Channel"], how="left")

    # Fill missing
    cpa_df["LC_Conversions"] = cpa_df["LC_Conversions"].fillna(0)
    cpa_df["Markov_Attribution(%)"] = cpa_df["Markov_Attribution(%)"].fillna(0)
    cpa_df["Shapley_Attribution(%)"] = cpa_df["Shapley_Attribution(%)"].fillna(0)

    # Map total conversions per brand
    cpa_df["Total_Brand_Conv"] = cpa_df["Brand_ID"].map(total_conversions_by_brand)

    # Compute attributed conversions for Markov and Shapley
    cpa_df["Markov_Conversions"] = (
        cpa_df["Total_Brand_Conv"] * cpa_df["Markov_Attribution(%)"] / 100
    )
    cpa_df["Shapley_Conversions"] = (
        cpa_df["Total_Brand_Conv"] * cpa_df["Shapley_Attribution(%)"] / 100
    )

    # Compute CPA for each model
    cpa_df["CPA_LastClick"] = np.where(
        cpa_df["LC_Conversions"] > 0,
        cpa_df["Budget"] / cpa_df["LC_Conversions"],
        np.inf,
    )
    cpa_df["CPA_Markov"] = np.where(
        cpa_df["Markov_Conversions"] > 0,
        cpa_df["Budget"] / cpa_df["Markov_Conversions"],
        np.inf,
    )
    cpa_df["CPA_Shapley"] = np.where(
        cpa_df["Shapley_Conversions"] > 0,
        cpa_df["Budget"] / cpa_df["Shapley_Conversions"],
        np.inf,
    )

    # CPA shift: how different is true CPA from last-click CPA?
    cpa_df["CPA_Shift_Markov(%)"] = np.where(
        (cpa_df["CPA_LastClick"] > 0) & (cpa_df["CPA_LastClick"] != np.inf),
        (cpa_df["CPA_Markov"] - cpa_df["CPA_LastClick"]) / cpa_df["CPA_LastClick"] * 100,
        0,
    )

    # Sort
    cpa_df = cpa_df.sort_values(["Brand_ID", "Channel"]).reset_index(drop=True)

    # Print summary
    for brand in sorted(cpa_df["Brand_ID"].unique()):
        brand_df = cpa_df[cpa_df["Brand_ID"] == brand]
        total_conv = brand_df["Total_Brand_Conv"].iloc[0]
        total_spend = brand_df["Budget"].sum()
        print(f"\n  --- {brand} ({total_conv:.0f} conversions, "
              f"Rs.{total_spend:,.0f} budget) ---")
        print(f"  {'Channel':<20s} {'CPA_LC':>12} {'CPA_Markov':>12} "
              f"{'CPA_Shapley':>12} {'Shift%':>8}")
        print("  " + "-" * 68)

        for _, row in brand_df.iterrows():
            def fmt_cpa(v):
                if v == np.inf or v > 1e8:
                    return "       inf"
                return f"Rs.{v:>8,.0f}"

            shift = row["CPA_Shift_Markov(%)"]
            shift_str = f"{shift:>+7.0f}%" if abs(shift) < 1e6 else "     N/A"

            print(f"  {row['Channel']:<20s} {fmt_cpa(row['CPA_LastClick'])} "
                  f"{fmt_cpa(row['CPA_Markov'])} "
                  f"{fmt_cpa(row['CPA_Shapley'])} "
                  f"{shift_str}")

    return cpa_df


# ---------------------------------------------------------------------------
# Phase 4C: Ad Fatigue / Saturation Curve Fitting
# ---------------------------------------------------------------------------
def _saturation_func(spend, a, b, c):
    """
    Hill / saturation function:
      conversions = a * spend^b / (c^b + spend^b)

    Parameters: a = max output, b = steepness, c = half-saturation spend
    """
    return a * np.power(spend, b) / (np.power(c, b) + np.power(spend, b))


def _log_func(spend, a, b):
    """
    Log response function:
      conversions = a * ln(spend + 1) + b
    """
    return a * np.log(spend + 1) + b


def fit_saturation_curves(
    df_clean: pd.DataFrame,
    df_spend: pd.DataFrame,
) -> dict:
    """
    Fit response (saturation) curves for each brand x channel to model
    the relationship between spend and conversions.

    This is used to detect ad fatigue / diminishing returns and later
    for budget optimization in Phase 5.

    Approach:
      - Divide the quarter into weekly time buckets
      - For each week, compute cumulative spend and cumulative conversions
      - Fit a log response curve to the data points

    Parameters
    ----------
    df_clean : pd.DataFrame
        Clean touchpoints.
    df_spend : pd.DataFrame
        Campaign spend.

    Returns
    -------
    dict : {(brand, channel): {
        "params": fitted parameters,
        "func": callable response function,
        "data": DataFrame of (spend, conversions) points,
        "model": "log" or "saturation",
        "r_squared": goodness of fit
    }}
    """
    print("\n" + "=" * 70)
    print("  PHASE 4C: AD FATIGUE / SATURATION CURVES")
    print("=" * 70)

    if df_clean["Timestamp"].dtype == object:
        df_clean = df_clean.copy()
        df_clean["Timestamp"] = pd.to_datetime(df_clean["Timestamp"])

    # Create weekly buckets
    df_clean = df_clean.copy()
    df_clean["Week"] = df_clean["Timestamp"].dt.isocalendar().week.astype(int)

    # Get spending rate per campaign (total budget / number of weeks)
    min_week = df_clean["Week"].min()
    max_week = df_clean["Week"].max()
    num_weeks = max_week - min_week + 1

    curves = {}

    brands = sorted(df_clean["Brand_ID"].unique())
    channels = sorted(df_clean["Channel"].unique())

    for brand in brands:
        for channel in channels:
            # Filter data
            mask = (df_clean["Brand_ID"] == brand) & (df_clean["Channel"] == channel)
            bc_data = df_clean[mask]

            if len(bc_data) == 0:
                continue

            # Get budget for this brand-channel
            spend_row = df_spend[
                (df_spend["Brand_ID"] == brand) & (df_spend["Channel"] == channel)
            ]
            if len(spend_row) == 0:
                continue

            total_budget = spend_row["Total_Budget_Allocated"].values[0]
            weekly_spend = total_budget / num_weeks

            # Weekly conversions
            weekly_conv = (
                bc_data[bc_data["Event_Type"] == "Purchase"]
                .groupby("Week")
                .size()
                .reindex(range(min_week, max_week + 1), fill_value=0)
            )

            # Build cumulative data points
            cum_spend = np.array([(i + 1) * weekly_spend
                                  for i in range(num_weeks)])
            cum_conv = np.cumsum(weekly_conv.values[:num_weeks])

            if cum_conv[-1] == 0:
                # No conversions for this brand-channel
                curves[(brand, channel)] = {
                    "params": (0, 0),
                    "func": lambda x: np.zeros_like(x),
                    "data": pd.DataFrame({"spend": cum_spend, "conversions": cum_conv}),
                    "model": "none",
                    "r_squared": 0,
                }
                continue

            # Fit log response: conv = a * ln(spend + 1) + b
            try:
                popt, _ = curve_fit(
                    _log_func,
                    cum_spend,
                    cum_conv.astype(float),
                    p0=[cum_conv[-1] / np.log(cum_spend[-1] + 1), 0],
                    maxfev=5000,
                )

                # R-squared
                predicted = _log_func(cum_spend, *popt)
                ss_res = np.sum((cum_conv - predicted) ** 2)
                ss_tot = np.sum((cum_conv - np.mean(cum_conv)) ** 2)
                r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                def make_func(p):
                    return lambda x: _log_func(x, *p)

                curves[(brand, channel)] = {
                    "params": tuple(popt),
                    "func": make_func(popt),
                    "data": pd.DataFrame({
                        "spend": cum_spend,
                        "conversions": cum_conv,
                    }),
                    "model": "log",
                    "r_squared": r_sq,
                }

            except Exception:
                # Fallback: linear model
                if cum_spend[-1] > 0:
                    slope = cum_conv[-1] / cum_spend[-1]
                else:
                    slope = 0
                curves[(brand, channel)] = {
                    "params": (slope,),
                    "func": lambda x, s=slope: s * x,
                    "data": pd.DataFrame({
                        "spend": cum_spend,
                        "conversions": cum_conv,
                    }),
                    "model": "linear",
                    "r_squared": 0,
                }

    # Print summary
    print(f"\n  Fitted {len(curves)} response curves")
    print(f"\n  {'Brand':<6} {'Channel':<20s} {'Model':>8} {'R-sq':>6} "
          f"{'Total Conv':>10} {'Budget':>16}")
    print("  " + "-" * 70)
    for (brand, channel), info in sorted(curves.items()):
        total_conv = info["data"]["conversions"].iloc[-1] if len(info["data"]) > 0 else 0
        total_spend = info["data"]["spend"].iloc[-1] if len(info["data"]) > 0 else 0
        print(f"  {brand:<6} {channel:<20s} {info['model']:>8} "
              f"{info['r_squared']:>5.3f} "
              f"{total_conv:>10,.0f} Rs.{total_spend:>13,.0f}")

    # Identify channels with strongest diminishing returns
    print("\n  DIMINISHING RETURNS ANALYSIS:")
    print("  (Marginal CPA at 25% vs 100% of budget)")
    print(f"  {'Brand':<6} {'Channel':<20s} {'CPA@25%':>12} {'CPA@100%':>12} "
          f"{'Fatigue Ratio':>14}")
    print("  " + "-" * 68)

    for (brand, channel), info in sorted(curves.items()):
        if info["model"] == "none" or info["r_squared"] < 0.5:
            continue
        data = info["data"]
        total_spend = data["spend"].iloc[-1]
        if total_spend == 0:
            continue

        # Marginal conversions at 25% and 100% of spend
        spend_25 = total_spend * 0.25
        spend_100 = total_spend

        func = info["func"]
        conv_at_25 = func(np.array([spend_25]))[0]
        conv_at_100 = func(np.array([spend_100]))[0]

        # Marginal CPA
        cpa_25 = spend_25 / conv_at_25 if conv_at_25 > 0 else np.inf
        cpa_100 = spend_100 / conv_at_100 if conv_at_100 > 0 else np.inf

        fatigue = cpa_100 / cpa_25 if cpa_25 > 0 and cpa_25 != np.inf else 0

        def fmt(v):
            return f"Rs.{v:>9,.0f}" if v != np.inf else "       inf"

        print(f"  {brand:<6} {channel:<20s} {fmt(cpa_25)} {fmt(cpa_100)} "
              f"{fatigue:>13.2f}x")

    return curves


# ---------------------------------------------------------------------------
# Master Phase 4 runner
# ---------------------------------------------------------------------------
def run_phase4(
    df_clean: pd.DataFrame,
    df_spend: pd.DataFrame,
    df_lc_financials: pd.DataFrame,
    df_markov: pd.DataFrame,
    df_shapley: pd.DataFrame,
) -> dict:
    """
    Execute the complete Phase 4 pipeline.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Clean touchpoints.
    df_spend : pd.DataFrame
        Campaign spend.
    df_lc_financials : pd.DataFrame
        Last-click financials from Phase 2.
    df_markov : pd.DataFrame
        Markov attribution from Phase 3.
    df_shapley : pd.DataFrame
        Shapley attribution from Phase 3.

    Returns
    -------
    dict with keys:
        "cost_verification" : cost validation DataFrame
        "cpa_comparison"    : CPA under all three models
        "response_curves"   : fitted saturation curves dict
    """
    print("\n" + "#" * 70)
    print("  ROI LENS -- PHASE 4: FINANCIAL LAYER")
    print("#" * 70)

    # 4A: Cost verification
    cost_df = verify_costs(df_clean, df_spend)

    # Get total conversions per brand
    purchases = df_clean[df_clean["Event_Type"] == "Purchase"]
    total_conv_by_brand = purchases.groupby("Brand_ID").size()

    # 4B: True CPA comparison
    cpa_df = compute_true_cpa(
        df_spend, df_lc_financials, df_markov, df_shapley,
        total_conv_by_brand
    )

    # 4C: Saturation / ad fatigue curves
    curves = fit_saturation_curves(df_clean, df_spend)

    # Summary
    print("\n" + "=" * 70)
    print("  PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"  Cost verification : {len(cost_df)} campaigns checked")
    print(f"  CPA comparison    : {len(cpa_df)} rows (brand x channel)")
    print(f"  Response curves   : {len(curves)} fitted")

    return {
        "cost_verification": cost_df,
        "cpa_comparison": cpa_df,
        "response_curves": curves,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_all
    from src.data_cleaner import clean_data
    from src.funnel_analysis import run_phase2
    from src.attribution import run_phase3

    # Load and clean
    print("Loading and cleaning data ...")
    df_tp, df_up, df_cs, df_merged = load_all()
    df_clean, bot_report = clean_data(
        df_merged, run_timestamp_check=False, verbose=False
    )

    # Run Phase 2
    print("\nRunning Phase 2 ...")
    phase2 = run_phase2(df_clean, df_cs)

    # Run Phase 3
    print("\nRunning Phase 3 ...")
    phase3 = run_phase3(df_clean, phase2["attr_agg"])

    # Run Phase 4
    phase4 = run_phase4(
        df_clean, df_cs,
        phase2["financials"],
        phase3["markov"],
        phase3["shapley"],
    )

    # Save outputs
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase4["cost_verification"].to_csv(
        out_dir / "cost_verification.csv", index=False
    )
    phase4["cpa_comparison"].to_csv(
        out_dir / "cpa_comparison.csv", index=False
    )

    # Save response curve data
    curve_data_rows = []
    for (brand, channel), info in phase4["response_curves"].items():
        for _, row in info["data"].iterrows():
            curve_data_rows.append({
                "Brand_ID": brand,
                "Channel": channel,
                "Cumulative_Spend": row["spend"],
                "Cumulative_Conversions": row["conversions"],
                "Model": info["model"],
                "R_Squared": info["r_squared"],
            })
    pd.DataFrame(curve_data_rows).to_csv(
        out_dir / "response_curves.csv", index=False
    )

    print(f"\n[Phase 4] All outputs saved to {out_dir}")
