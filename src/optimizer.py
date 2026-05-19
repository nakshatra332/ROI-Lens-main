"""
optimizer.py -- Phase 5: Constrained Budget Optimization
==========================================================
Allocates Rs.10 Crore per brand (Rs.100 Crore total) to maximize
overall conversions, using the response curves from Phase 4 and
the attribution insights from Phase 3.

Uses scipy.optimize.minimize (SLSQP) with non-linear response
functions that naturally capture ad fatigue / diminishing returns.

Constraints:
  (1) Total spend per brand = Rs.10 Crore exactly
  (2) Minimum spend per channel = 5% of brand budget (viability floor)
  (3) Maximum spend per channel = 50% of brand budget (diversification cap)
  (4) Non-negativity
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRAND_BUDGET = 1e8            # Rs.10 Crore per brand
MIN_SHARE = 0.05              # 5% minimum per channel
MAX_SHARE = 0.50              # 50% maximum per channel
NUM_BRANDS = 10
TOTAL_BUDGET = BRAND_BUDGET * NUM_BRANDS  # Rs.100 Crore


# ---------------------------------------------------------------------------
# Response function (must match Phase 4 fitting)
# ---------------------------------------------------------------------------
def _log_response(spend, a, b):
    """Log response: conversions = a * ln(spend + 1) + b"""
    return a * np.log(spend + 1) + b


# ---------------------------------------------------------------------------
# Phase 5A: Single brand optimizer
# ---------------------------------------------------------------------------
def optimize_single_brand(
    brand_id: str,
    channels: list,
    response_curves: dict,
    budget: float = BRAND_BUDGET,
    verbose: bool = True,
) -> dict:
    """
    Optimize budget allocation for a single brand to maximize
    total conversions across all channels.

    Parameters
    ----------
    brand_id : str
        Brand identifier (e.g., "B01").
    channels : list
        List of channel names.
    response_curves : dict
        Fitted response curves from Phase 4.
        Keys are (brand_id, channel) tuples.
    budget : float
        Total budget for this brand (default Rs.10 Cr).
    verbose : bool
        Print optimization details.

    Returns
    -------
    dict with keys:
        "brand_id"          : str
        "allocation"        : dict {channel: spend}
        "expected_conv"     : dict {channel: conversions}
        "total_conv"        : float
        "success"           : bool
    """
    n = len(channels)
    min_spend = budget * MIN_SHARE
    max_spend = budget * MAX_SHARE

    # Build the objective function (negative because we minimize)
    def neg_total_conversions(spends):
        total = 0.0
        for i, ch in enumerate(channels):
            key = (brand_id, ch)
            if key in response_curves:
                info = response_curves[key]
                if info["model"] == "log" and info["r_squared"] > 0:
                    params = info["params"]
                    conv = _log_response(spends[i], *params)
                    total += max(conv, 0)  # Prevent negative conversions
                elif info["model"] == "linear":
                    total += info["params"][0] * spends[i]
            # If no curve, contribution is 0
        return -total

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - budget}  # Sum = budget
    ]

    # Bounds: [min_spend, max_spend] per channel
    bounds = [(min_spend, max_spend) for _ in channels]

    # Initial guess: equal split
    x0 = np.array([budget / n] * n)

    # Optimize
    result = minimize(
        neg_total_conversions,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    # Extract results
    allocation = {ch: result.x[i] for i, ch in enumerate(channels)}
    expected_conv = {}
    for i, ch in enumerate(channels):
        key = (brand_id, ch)
        if key in response_curves:
            info = response_curves[key]
            if info["model"] == "log" and info["r_squared"] > 0:
                conv = _log_response(result.x[i], *info["params"])
                expected_conv[ch] = max(conv, 0)
            elif info["model"] == "linear":
                expected_conv[ch] = info["params"][0] * result.x[i]
            else:
                expected_conv[ch] = 0
        else:
            expected_conv[ch] = 0

    total_conv = sum(expected_conv.values())

    if verbose:
        print(f"\n  {brand_id}: Optimization {'SUCCESS' if result.success else 'FAILED'}")
        print(f"    {'Channel':<20s} {'Allocation':>16} {'Share%':>8} "
              f"{'Exp Conv':>10}")
        print("    " + "-" * 58)
        for ch in channels:
            share = allocation[ch] / budget * 100
            print(f"    {ch:<20s} Rs.{allocation[ch]:>13,.0f} "
                  f"{share:>7.1f}% {expected_conv[ch]:>10.1f}")
        print(f"    {'TOTAL':<20s} Rs.{budget:>13,.0f} "
              f"{'100.0%':>8} {total_conv:>10.1f}")

    return {
        "brand_id": brand_id,
        "allocation": allocation,
        "expected_conv": expected_conv,
        "total_conv": total_conv,
        "success": result.success,
    }


# ---------------------------------------------------------------------------
# Phase 5B: All brands optimizer
# ---------------------------------------------------------------------------
def optimize_all_brands(
    response_curves: dict,
    df_spend: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Optimize budget allocation for all 10 brands.

    Returns
    -------
    pd.DataFrame with columns:
        Brand_ID, Channel, Current_Spend, Optimized_Spend, Delta_Spend,
        Delta_Spend(%), Current_Conv, Expected_Conv, Conv_Lift(%)
    """
    print("\n" + "=" * 70)
    print("  PHASE 5B: BUDGET OPTIMIZATION (ALL BRANDS)")
    print("=" * 70)
    print(f"\n  Budget per brand : Rs.{BRAND_BUDGET:,.0f} (Rs.{BRAND_BUDGET/1e7:.0f} Cr)")
    print(f"  Total budget     : Rs.{TOTAL_BUDGET:,.0f} (Rs.{TOTAL_BUDGET/1e7:.0f} Cr)")
    print(f"  Channel bounds   : [{MIN_SHARE*100:.0f}%, {MAX_SHARE*100:.0f}%]")

    # Current spend per brand x channel
    current_spend = (
        df_spend.groupby(["Brand_ID", "Channel"])["Total_Budget_Allocated"]
        .sum()
        .reset_index()
        .rename(columns={"Total_Budget_Allocated": "Current_Spend"})
    )

    # Current conversions per brand x channel
    current_conv = (
        df_clean[df_clean["Event_Type"] == "Purchase"]
        .groupby(["Brand_ID", "Channel"])
        .size()
        .reset_index(name="Current_Conv")
    )

    brands = sorted(df_spend["Brand_ID"].unique())
    channels = sorted(df_spend["Channel"].unique())

    all_results = []

    for brand in brands:
        result = optimize_single_brand(
            brand, channels, response_curves, BRAND_BUDGET, verbose=True
        )

        for ch in channels:
            all_results.append({
                "Brand_ID": brand,
                "Channel": ch,
                "Optimized_Spend": result["allocation"].get(ch, 0),
                "Expected_Conv": result["expected_conv"].get(ch, 0),
            })

    opt_df = pd.DataFrame(all_results)

    # Merge with current data
    opt_df = opt_df.merge(current_spend, on=["Brand_ID", "Channel"], how="left")
    opt_df = opt_df.merge(current_conv, on=["Brand_ID", "Channel"], how="left")
    opt_df["Current_Spend"] = opt_df["Current_Spend"].fillna(0)
    opt_df["Current_Conv"] = opt_df["Current_Conv"].fillna(0)

    # Compute deltas
    opt_df["Delta_Spend"] = opt_df["Optimized_Spend"] - opt_df["Current_Spend"]
    opt_df["Delta_Spend(%)"] = np.where(
        opt_df["Current_Spend"] > 0,
        opt_df["Delta_Spend"] / opt_df["Current_Spend"] * 100,
        0,
    )
    opt_df["Conv_Lift(%)"] = np.where(
        opt_df["Current_Conv"] > 0,
        (opt_df["Expected_Conv"] - opt_df["Current_Conv"]) / opt_df["Current_Conv"] * 100,
        0,
    )

    # Reorder columns
    opt_df = opt_df[[
        "Brand_ID", "Channel",
        "Current_Spend", "Optimized_Spend", "Delta_Spend", "Delta_Spend(%)",
        "Current_Conv", "Expected_Conv", "Conv_Lift(%)",
    ]]

    opt_df = opt_df.sort_values(["Brand_ID", "Channel"]).reset_index(drop=True)

    return opt_df


# ---------------------------------------------------------------------------
# Phase 5C: Reallocation summary & recommendations
# ---------------------------------------------------------------------------
def print_reallocation_summary(opt_df: pd.DataFrame):
    """Print a formatted reallocation summary with recommendations."""
    print("\n" + "=" * 70)
    print("  PHASE 5C: REALLOCATION SUMMARY")
    print("=" * 70)

    for brand in sorted(opt_df["Brand_ID"].unique()):
        brand_df = opt_df[opt_df["Brand_ID"] == brand]
        total_current = brand_df["Current_Spend"].sum()
        total_opt = brand_df["Optimized_Spend"].sum()
        total_current_conv = brand_df["Current_Conv"].sum()
        total_exp_conv = brand_df["Expected_Conv"].sum()

        print(f"\n  === {brand} ===")
        print(f"  {'Channel':<20s} {'Current':>14} {'Optimized':>14} "
              f"{'Delta':>14} {'Curr Conv':>10} {'Exp Conv':>10}")
        print("  " + "-" * 86)

        for _, row in brand_df.iterrows():
            delta_str = f"{row['Delta_Spend']:>+14,.0f}"
            print(f"  {row['Channel']:<20s} "
                  f"Rs.{row['Current_Spend']:>11,.0f} "
                  f"Rs.{row['Optimized_Spend']:>11,.0f} "
                  f"{delta_str} "
                  f"{row['Current_Conv']:>10.0f} "
                  f"{row['Expected_Conv']:>10.1f}")

        conv_lift = ((total_exp_conv - total_current_conv) / total_current_conv * 100
                     if total_current_conv > 0 else 0)
        print(f"  {'TOTAL':<20s} "
              f"Rs.{total_current:>11,.0f} "
              f"Rs.{total_opt:>11,.0f} "
              f"{'':>14} "
              f"{total_current_conv:>10.0f} "
              f"{total_exp_conv:>10.1f}")
        print(f"  Expected conversion lift: {conv_lift:+.1f}%")

    # Overall summary
    print("\n" + "=" * 70)
    print("  OVERALL REALLOCATION IMPACT")
    print("=" * 70)

    total_current_conv = opt_df["Current_Conv"].sum()
    total_exp_conv = opt_df["Expected_Conv"].sum()
    overall_lift = ((total_exp_conv - total_current_conv) / total_current_conv * 100
                    if total_current_conv > 0 else 0)

    print(f"\n  Current total conversions  : {total_current_conv:>10,.0f}")
    print(f"  Expected total conversions : {total_exp_conv:>10,.1f}")
    print(f"  Overall conversion lift    : {overall_lift:>+10.1f}%")
    print(f"  Total budget               : Rs.{TOTAL_BUDGET:>14,.0f} "
          f"(Rs.{TOTAL_BUDGET/1e7:.0f} Cr)")

    # Top channels to INCREASE spend
    increases = opt_df[opt_df["Delta_Spend"] > 0].nlargest(5, "Delta_Spend")
    print("\n  TOP 5 CHANNELS TO INCREASE SPEND:")
    for _, row in increases.iterrows():
        print(f"    {row['Brand_ID']} {row['Channel']:<20s}: "
              f"Rs.{row['Delta_Spend']:>+14,.0f}")

    # Top channels to DECREASE spend
    decreases = opt_df[opt_df["Delta_Spend"] < 0].nsmallest(5, "Delta_Spend")
    print("\n  TOP 5 CHANNELS TO DECREASE SPEND:")
    for _, row in decreases.iterrows():
        print(f"    {row['Brand_ID']} {row['Channel']:<20s}: "
              f"Rs.{row['Delta_Spend']:>+14,.0f}")

    # Channels to consider DEFUNDING (hitting minimum cap)
    defund = opt_df[
        opt_df["Optimized_Spend"] <= BRAND_BUDGET * MIN_SHARE * 1.01
    ].copy()
    if len(defund) > 0:
        print(f"\n  CHANNELS AT MINIMUM SPEND (candidates for defunding):")
        for _, row in defund.iterrows():
            print(f"    {row['Brand_ID']} {row['Channel']:<20s}: "
                  f"Rs.{row['Optimized_Spend']:>11,.0f} "
                  f"(was Rs.{row['Current_Spend']:>11,.0f})")

    # Channels to consider FREQUENCY CAP (hitting maximum cap)
    freq_cap = opt_df[
        opt_df["Optimized_Spend"] >= BRAND_BUDGET * MAX_SHARE * 0.99
    ].copy()
    if len(freq_cap) > 0:
        print(f"\n  CHANNELS AT MAXIMUM SPEND (may need frequency caps):")
        for _, row in freq_cap.iterrows():
            print(f"    {row['Brand_ID']} {row['Channel']:<20s}: "
                  f"Rs.{row['Optimized_Spend']:>11,.0f} "
                  f"({row['Expected_Conv']:.0f} conv)")


# ---------------------------------------------------------------------------
# Phase 5D: Sensitivity analysis
# ---------------------------------------------------------------------------
def sensitivity_analysis(
    response_curves: dict,
    df_spend: pd.DataFrame,
    base_budget: float = BRAND_BUDGET,
) -> pd.DataFrame:
    """
    Analyze how total conversions change with budget variations.
    Tests +/-20% and +/-40% budget scenarios.

    Returns
    -------
    pd.DataFrame with scenario results.
    """
    print("\n" + "=" * 70)
    print("  PHASE 5D: SENSITIVITY ANALYSIS")
    print("=" * 70)

    scenarios = [0.6, 0.8, 1.0, 1.2, 1.4]
    brands = sorted(df_spend["Brand_ID"].unique())
    channels = sorted(df_spend["Channel"].unique())

    results = []
    for multiplier in scenarios:
        test_budget = base_budget * multiplier
        total_conv = 0
        for brand in brands:
            result = optimize_single_brand(
                brand, channels, response_curves, test_budget, verbose=False
            )
            total_conv += result["total_conv"]

        results.append({
            "Budget_Multiplier": multiplier,
            "Budget_Per_Brand": test_budget,
            "Total_Budget": test_budget * NUM_BRANDS,
            "Total_Conversions": total_conv,
        })

    df_sens = pd.DataFrame(results)

    # Marginal return
    df_sens["Marginal_Conv"] = df_sens["Total_Conversions"].diff()
    df_sens["Marginal_Budget"] = df_sens["Total_Budget"].diff()
    df_sens["Marginal_CPA"] = np.where(
        df_sens["Marginal_Conv"] > 0,
        df_sens["Marginal_Budget"] / df_sens["Marginal_Conv"],
        np.inf,
    )

    print(f"\n  {'Budget':>12} {'Per Brand':>14} {'Total Conv':>12} "
          f"{'Marginal CPA':>14}")
    print("  " + "-" * 56)
    for _, row in df_sens.iterrows():
        cpa_str = (f"Rs.{row['Marginal_CPA']:>11,.0f}"
                   if row["Marginal_CPA"] != np.inf and not np.isnan(row["Marginal_CPA"])
                   else "           -")
        print(f"  Rs.{row['Total_Budget']:>10,.0f} "
              f"Rs.{row['Budget_Per_Brand']:>12,.0f} "
              f"{row['Total_Conversions']:>12,.1f} "
              f"{cpa_str}")

    return df_sens


# ---------------------------------------------------------------------------
# Master Phase 5 runner
# ---------------------------------------------------------------------------
def run_phase5(
    response_curves: dict,
    df_spend: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> dict:
    """
    Execute the complete Phase 5 pipeline.

    Parameters
    ----------
    response_curves : dict
        Fitted response curves from Phase 4.
    df_spend : pd.DataFrame
        Campaign spend data.
    df_clean : pd.DataFrame
        Clean touchpoints.

    Returns
    -------
    dict with keys:
        "optimization"  : Reallocation DataFrame
        "sensitivity"   : Sensitivity analysis DataFrame
    """
    print("\n" + "#" * 70)
    print("  ROI LENS -- PHASE 5: BUDGET OPTIMIZATION")
    print("#" * 70)

    # 5A+5B: Optimize all brands
    opt_df = optimize_all_brands(response_curves, df_spend, df_clean)

    # 5C: Reallocation summary
    print_reallocation_summary(opt_df)

    # 5D: Sensitivity analysis
    sens_df = sensitivity_analysis(response_curves, df_spend)

    # Summary
    print("\n" + "=" * 70)
    print("  PHASE 5 COMPLETE")
    print("=" * 70)
    print(f"  Optimization rows : {len(opt_df)} (brand x channel)")
    print(f"  Sensitivity cases : {len(sens_df)} scenarios")

    return {
        "optimization": opt_df,
        "sensitivity": sens_df,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_all
    from src.data_cleaner import clean_data
    from src.funnel_analysis import run_phase2
    from src.attribution import run_phase3
    from src.financials import run_phase4

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
    print("\nRunning Phase 4 ...")
    phase4 = run_phase4(
        df_clean, df_cs,
        phase2["financials"], phase3["markov"], phase3["shapley"]
    )

    # Run Phase 5
    phase5 = run_phase5(phase4["response_curves"], df_cs, df_clean)

    # Save outputs
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase5["optimization"].to_csv(
        out_dir / "budget_optimization.csv", index=False
    )
    phase5["sensitivity"].to_csv(
        out_dir / "sensitivity_analysis.csv", index=False
    )

    print(f"\n[Phase 5] All outputs saved to {out_dir}")
