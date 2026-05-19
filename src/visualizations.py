"""
visualizations.py -- Phase 6: Chart Generation
================================================
Generates all publication-quality charts from the Phase 1-5 outputs.
Saves figures to outputs/figures/ for use in the notebook and slides.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
PALETTE = {
    "blue":     "#2563EB",
    "teal":     "#0D9488",
    "orange":   "#EA580C",
    "red":      "#DC2626",
    "green":    "#16A34A",
    "purple":   "#7C3AED",
    "slate":    "#475569",
    "sky":      "#0EA5E9",
    "amber":    "#D97706",
    "rose":     "#E11D48",
}
CHANNEL_COLORS = {
    "Google Search":    "#4285F4",
    "Instagram":        "#E1306C",
    "YouTube":          "#FF0000",
    "Influencer Blog":  "#06B6D4",
    "Marketplace":      "#F59E0B",
}

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "outputs" / "results"


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor":   "#FAFAFA",
        "axes.edgecolor":   "#D1D5DB",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.color":       "#9CA3AF",
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.titlesize":   14,
        "axes.titleweight": "bold",
        "axes.labelsize":   12,
        "legend.fontsize":  10,
        "figure.dpi":       150,
    })


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _fmt_inr(val, _=None):
    """Format INR values for axis labels."""
    sign = "-" if val < 0 else ""
    v = abs(val)
    if v >= 1e7:
        return f"{sign}Rs.{v/1e7:.0f}Cr"
    if v >= 1e5:
        return f"{sign}Rs.{v/1e5:.0f}L"
    if v == 0:
        return "0"
    return f"{sign}Rs.{v:,.0f}"


# ---------------------------------------------------------------------------
# Chart 1: Bot Detection
# ---------------------------------------------------------------------------
def plot_bot_detection():
    """Bar chart showing bot traffic impact per channel."""
    _setup_style()
    # Hardcoded from Phase 1 output (clean vs raw)
    channels = ["YouTube", "Instagram", "Marketplace", "Google Search", "Influencer Blog"]
    bot_pct = [24.3, 24.1, 24.1, 23.7, 23.8]
    clean_pct = [100 - b for b in bot_pct]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = range(len(channels))
    bars_clean = ax.barh(y, clean_pct, color=PALETTE["teal"], label="Clean Traffic", height=0.6)
    bars_bot = ax.barh(y, bot_pct, left=clean_pct, color=PALETTE["red"], label="Bot Traffic", height=0.6, alpha=0.85)

    for i, (c, b) in enumerate(zip(clean_pct, bot_pct)):
        ax.text(c + b/2, i, f"{b:.1f}%", ha="center", va="center", fontweight="bold", color="white", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(channels)
    ax.set_xlabel("Traffic Share (%)")
    ax.set_title("Bot Traffic Detection: 1,371 Users Generated 24% of All Events")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "01_bot_detection")


# ---------------------------------------------------------------------------
# Chart 2: Conversion Funnel
# ---------------------------------------------------------------------------
def plot_funnel():
    """Funnel chart showing event progression."""
    _setup_style()
    df = pd.read_csv(DATA_DIR / "funnel_brand.csv")
    stages = ["Impressions", "Clicks", "Add_to_Cart", "Purchases"]
    labels = ["Impressions", "Clicks", "Add-to-Cart", "Purchases"]
    totals = [df[s].sum() for s in stages]

    colors = ["#3B82F6", "#0EA5E9", "#14B8A6", "#10B981"]
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, totals, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + totals[0]*0.01,
                f"{val:,.0f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    # Drop-off annotations
    for i in range(1, len(totals)):
        drop = (1 - totals[i]/totals[i-1]) * 100
        mid_x = i - 0.5
        mid_y = (totals[i-1] + totals[i]) / 2
        ax.annotate(f"-{drop:.0f}%", xy=(mid_x, mid_y), fontsize=10,
                    color=PALETTE["red"], fontweight="bold", ha="center")

    ax.set_title("Conversion Funnel: Impression to Purchase (All Brands)")
    ax.set_ylabel("Event Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    fig.tight_layout()
    _save(fig, "02_conversion_funnel")


# ---------------------------------------------------------------------------
# Chart 3: Brand Funnel Comparison
# ---------------------------------------------------------------------------
def plot_brand_funnel():
    _setup_style()
    df = pd.read_csv(DATA_DIR / "funnel_brand.csv")
    df = df.sort_values("Purchases", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["blue"] if p < 500 else PALETTE["teal"] if p < 1000 else PALETTE["green"]
              for p in df["Purchases"]]

    bars = ax.barh(df["Brand_ID"], df["Purchases"], color=colors, height=0.6, edgecolor="white")
    for bar, val in zip(bars, df["Purchases"]):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontweight="bold", fontsize=11)

    ax.set_xlabel("Total Purchases")
    ax.set_title("Conversions by Brand: B07 and B02 Lead the Pack")
    fig.tight_layout()
    _save(fig, "03_brand_conversions")


# ---------------------------------------------------------------------------
# Chart 4: Attribution Comparison (per brand)
# ---------------------------------------------------------------------------
def plot_attribution_comparison():
    _setup_style()
    df = pd.read_csv(DATA_DIR / "attribution_comparison.csv")

    # Pick 3 most interesting brands: B01 (Instagram-heavy), B02 (Google-heavy), B07 (top converter)
    highlight_brands = ["B01", "B02", "B07"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, brand in zip(axes, highlight_brands):
        bdf = df[df["Brand_ID"] == brand].sort_values("Channel")
        x = range(len(bdf))
        w = 0.25

        ax.bar([i - w for i in x], bdf["LastClick(%)"], w, label="Last-Click", color=PALETTE["slate"])
        ax.bar(x, bdf["Markov(%)"], w, label="Markov", color=PALETTE["blue"])
        ax.bar([i + w for i in x], bdf["Shapley(%)"], w, label="Shapley", color=PALETTE["orange"])

        ax.set_xticks(x)
        short_names = [c.replace("Influencer Blog", "Influencer").replace("Google Search", "Google")
                       for c in bdf["Channel"]]
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{brand}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Attribution (%)" if brand == "B01" else "")

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Attribution Model Comparison: Last-Click vs Markov vs Shapley", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "04_attribution_comparison")


# ---------------------------------------------------------------------------
# Chart 5: Attribution Shift (Delta heatmap)
# ---------------------------------------------------------------------------
def plot_attribution_heatmap():
    _setup_style()
    df = pd.read_csv(DATA_DIR / "attribution_comparison.csv")
    pivot = df.pivot(index="Brand_ID", columns="Channel", values="Delta_Markov")
    pivot.columns = [c.replace("Influencer Blog", "Influencer").replace("Google Search", "Google") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt="+.1f", cmap="RdYlGn", center=0,
                linewidths=1.5, linecolor="white", ax=ax, cbar_kws={"label": "Delta (%)"})
    ax.set_title("Attribution Shift: Markov vs Last-Click\n(Green = Under-credited by LC, Red = Over-credited)")
    ax.set_ylabel("Brand")
    ax.set_xlabel("Channel")
    fig.tight_layout()
    _save(fig, "05_attribution_heatmap")


# ---------------------------------------------------------------------------
# Chart 6: Channel Roles
# ---------------------------------------------------------------------------
def plot_channel_roles():
    _setup_style()
    df = pd.read_csv(DATA_DIR / "channel_roles.csv")

    role_map = {"Primer/Introducer": 0, "Influencer/Assist": 1, "Closer/Converter": 2}
    role_colors = {0: "#3B82F6", 1: "#A78BFA", 2: "#10B981"}
    role_labels = {0: "Primer", 1: "Influencer", 2: "Closer"}

    pivot = df.pivot(index="Brand_ID", columns="Channel", values="Funnel_Role")
    pivot_num = pivot.map(lambda x: role_map.get(x, 1))
    pivot_num.columns = [c.replace("Influencer Blog", "Influencer").replace("Google Search", "Google") for c in pivot_num.columns]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot_num, annot=pivot.values, fmt="", cmap=["#3B82F6", "#A78BFA", "#10B981"],
                linewidths=2, linecolor="white", ax=ax, cbar=False, vmin=0, vmax=2)
    ax.set_title("Channel Funnel Roles by Brand\n(Blue=Primer, Purple=Influencer, Green=Closer)")
    ax.set_ylabel("Brand")
    fig.tight_layout()
    _save(fig, "06_channel_roles")


# ---------------------------------------------------------------------------
# Chart 7: CPA Comparison
# ---------------------------------------------------------------------------
def plot_cpa_comparison():
    _setup_style()
    df = pd.read_csv(DATA_DIR / "cpa_comparison.csv")

    # Brand-level avg CPA
    brand_cpa = df.groupby("Brand_ID").apply(
        lambda g: pd.Series({
            "CPA_LC": g["Budget"].sum() / g["LC_Conversions"].sum() if g["LC_Conversions"].sum() > 0 else 0,
            "CPA_Markov": g["Budget"].sum() / g["Markov_Conversions"].sum() if g["Markov_Conversions"].sum() > 0 else 0,
        })
    ).reset_index()
    brand_cpa = brand_cpa.sort_values("Brand_ID")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(brand_cpa))
    w = 0.35
    ax.bar([i - w/2 for i in x], brand_cpa["CPA_LC"], w, label="CPA (Last-Click)", color=PALETTE["slate"])
    ax.bar([i + w/2 for i in x], brand_cpa["CPA_Markov"], w, label="CPA (Markov)", color=PALETTE["blue"])

    ax.set_xticks(x)
    ax.set_xticklabels(brand_cpa["Brand_ID"])
    ax.set_ylabel("Cost Per Acquisition (Rs.)")
    ax.set_title("CPA Comparison: Last-Click vs True (Markov) Attribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    ax.legend()
    fig.tight_layout()
    _save(fig, "07_cpa_comparison")


# ---------------------------------------------------------------------------
# Chart 8: Budget Reallocation
# ---------------------------------------------------------------------------
def plot_budget_reallocation():
    _setup_style()
    df = pd.read_csv(DATA_DIR / "budget_optimization.csv")

    # Aggregate by channel across all brands
    ch_agg = df.groupby("Channel").agg(
        Current=("Current_Spend", "sum"),
        Optimized=("Optimized_Spend", "sum")
    ).reset_index()
    ch_agg["Delta"] = ch_agg["Optimized"] - ch_agg["Current"]
    ch_agg = ch_agg.sort_values("Delta")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE["green"] if d > 0 else PALETTE["red"] for d in ch_agg["Delta"]]
    bars = ax.barh(ch_agg["Channel"], ch_agg["Delta"], color=colors, height=0.6)

    for bar, val in zip(bars, ch_agg["Delta"]):
        sign = "+" if val > 0 else ""
        x_pos = bar.get_width() + (2e6 if val > 0 else -2e6)
        ha = "left" if val > 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{sign}Rs.{val/1e7:.1f}Cr", va="center", ha=ha, fontweight="bold", fontsize=11)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Budget Change (Rs.)")
    ax.set_title("Recommended Budget Shifts by Channel (Rs.100 Cr Reallocation)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_inr))
    fig.tight_layout()
    _save(fig, "08_budget_reallocation")


# ---------------------------------------------------------------------------
# Chart 9: Per-Brand Conversion Lift
# ---------------------------------------------------------------------------
def plot_conversion_lift():
    _setup_style()
    df = pd.read_csv(DATA_DIR / "budget_optimization.csv")

    brand_lift = df.groupby("Brand_ID").agg(
        Current=("Current_Conv", "sum"),
        Expected=("Expected_Conv", "sum"),
    ).reset_index()
    brand_lift["Lift(%)"] = (brand_lift["Expected"] - brand_lift["Current"]) / brand_lift["Current"] * 100
    brand_lift = brand_lift.sort_values("Lift(%)", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["green"] if l > 0 else PALETTE["red"] for l in brand_lift["Lift(%)"]]
    bars = ax.barh(brand_lift["Brand_ID"], brand_lift["Lift(%)"], color=colors, height=0.6)

    for bar, val in zip(bars, brand_lift["Lift(%)"]):
        x_pos = bar.get_width() + 0.5 if val > 0 else bar.get_width() - 0.5
        ha = "left" if val > 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", va="center", ha=ha, fontweight="bold", fontsize=11)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Expected Conversion Lift (%)")
    ax.set_title("Expected Conversion Lift After Budget Optimization (+17.4% Overall)")
    fig.tight_layout()
    _save(fig, "09_conversion_lift")


# ---------------------------------------------------------------------------
# Chart 10: Sensitivity Analysis
# ---------------------------------------------------------------------------
def plot_sensitivity():
    _setup_style()
    df = pd.read_csv(DATA_DIR / "sensitivity_analysis.csv")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df["Total_Budget"]/1e7, df["Total_Conversions"], "o-",
             color=PALETTE["blue"], linewidth=2.5, markersize=8, label="Total Conversions")
    ax1.set_xlabel("Total Budget (Rs. Cr)")
    ax1.set_ylabel("Expected Conversions", color=PALETTE["blue"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])

    # Highlight base case
    base = df[df["Budget_Multiplier"] == 1.0].iloc[0]
    ax1.annotate(f"Base: {base['Total_Conversions']:,.0f}",
                 xy=(base["Total_Budget"]/1e7, base["Total_Conversions"]),
                 xytext=(15, 15), textcoords="offset points",
                 fontweight="bold", fontsize=11,
                 arrowprops=dict(arrowstyle="->", color=PALETTE["blue"]))

    ax2 = ax1.twinx()
    marginal = df["Marginal_CPA"].dropna()
    budget_vals = df.loc[marginal.index, "Total_Budget"] / 1e7
    ax2.bar(budget_vals, marginal/1e5, width=1.5, alpha=0.3, color=PALETTE["orange"], label="Marginal CPA")
    ax2.set_ylabel("Marginal CPA (Rs. Lakhs)", color=PALETTE["orange"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["orange"])

    ax1.set_title("Sensitivity Analysis: Budget vs Conversions (Diminishing Returns)")
    fig.tight_layout()
    _save(fig, "10_sensitivity_analysis")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------
def generate_all_charts():
    """Generate all Phase 6 charts."""
    print("\n" + "=" * 70)
    print("  PHASE 6: GENERATING CHARTS")
    print("=" * 70 + "\n")

    plot_bot_detection()
    plot_funnel()
    plot_brand_funnel()
    plot_attribution_comparison()
    plot_attribution_heatmap()
    plot_channel_roles()
    plot_cpa_comparison()
    plot_budget_reallocation()
    plot_conversion_lift()
    plot_sensitivity()

    print(f"\n  All 10 charts saved to {FIG_DIR}")


if __name__ == "__main__":
    generate_all_charts()
