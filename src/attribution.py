"""
attribution.py -- Phase 3: Multi-Touch Attribution Engine
==========================================================
Implements two complementary probabilistic attribution models to
replace the flawed last-click baseline from Phase 2.

Models:
  1. Markov Chain Attribution
     - Builds a transition probability matrix from user journey paths
     - Computes "removal effects" for each channel
     - Attribution = normalized removal effects

  2. Shapley Value Attribution
     - Game-theoretic fair-credit allocation
     - Computes marginal contribution of each channel across all
       possible coalitions (2^5 = 32 subsets for 5 channels)

Also classifies each channel's funnel role:
  - Primer/Introducer   : first touch in most journeys
  - Influencer/Assist   : appears mid-journey, amplifies intent
  - Closer/Converter    : final touch before purchase
"""

import pandas as pd
import numpy as np
from itertools import combinations
from math import factorial
from collections import defaultdict
from pathlib import Path
import copy


# ---------------------------------------------------------------------------
# Phase 3A: Journey Path Construction
# ---------------------------------------------------------------------------
def build_journey_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct ordered channel journey paths for every user.

    For each user, sort touchpoints by timestamp, extract the sequence
    of channels visited, and mark whether the journey ended in a
    Purchase (converting) or not (non-converting).

    Consecutive duplicate channels are collapsed:
      e.g. Instagram -> Instagram -> Google -> Instagram
      becomes Instagram -> Google -> Instagram

    Parameters
    ----------
    df : pd.DataFrame
        Clean touchpoints with User_ID, Timestamp, Channel,
        Event_Type, Brand_ID.

    Returns
    -------
    pd.DataFrame with columns:
        User_ID, Brand_ID, Path (list of channels),
        Path_Str (string representation), Converted (bool),
        Path_Length (int)
    """
    print("\n" + "=" * 70)
    print("  PHASE 3A: JOURNEY PATH CONSTRUCTION")
    print("=" * 70)

    # Ensure Timestamp is datetime
    if df["Timestamp"].dtype == object:
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Sort all touchpoints
    df_sorted = df.sort_values(["User_ID", "Timestamp"])

    # Determine which users converted
    converters = set(
        df_sorted[df_sorted["Event_Type"] == "Purchase"]["User_ID"].unique()
    )

    journeys = []
    current_user = None
    current_brand = None
    current_path = []

    for _, row in df_sorted.iterrows():
        uid = row["User_ID"]
        if uid != current_user:
            # Save previous user's journey
            if current_user is not None and current_path:
                journeys.append({
                    "User_ID": current_user,
                    "Brand_ID": current_brand,
                    "Path": current_path,
                    "Converted": current_user in converters,
                })
            current_user = uid
            current_brand = row["Brand_ID"]
            current_path = []

        channel = row["Channel"]
        # Collapse consecutive duplicates
        if not current_path or current_path[-1] != channel:
            current_path.append(channel)

    # Don't forget the last user
    if current_user is not None and current_path:
        journeys.append({
            "User_ID": current_user,
            "Brand_ID": current_brand,
            "Path": current_path,
            "Converted": current_user in converters,
        })

    df_journeys = pd.DataFrame(journeys)
    df_journeys["Path_Str"] = df_journeys["Path"].apply(lambda p: " > ".join(p))
    df_journeys["Path_Length"] = df_journeys["Path"].apply(len)

    total = len(df_journeys)
    conv = df_journeys["Converted"].sum()
    avg_len = df_journeys["Path_Length"].mean()

    print(f"\n  Total journeys     : {total:,}")
    print(f"  Converting         : {conv:,} ({conv/total*100:.1f}%)")
    print(f"  Non-converting     : {total - conv:,} ({(total-conv)/total*100:.1f}%)")
    print(f"  Avg path length    : {avg_len:.1f} channels")
    print(f"  Max path length    : {df_journeys['Path_Length'].max()}")

    # Top 5 most common converting paths
    conv_paths = df_journeys[df_journeys["Converted"]]
    if len(conv_paths) > 0:
        print("\n  Top 5 converting paths:")
        top_paths = conv_paths["Path_Str"].value_counts().head(5)
        for path, cnt in top_paths.items():
            print(f"    {cnt:>5,}x : {path}")

    return df_journeys


# ---------------------------------------------------------------------------
# Phase 3B: Markov Chain Attribution
# ---------------------------------------------------------------------------
def _build_transition_matrix(df_journeys: pd.DataFrame, brand: str) -> dict:
    """
    Build a transition probability matrix for a specific brand.

    States: Start, [channels...], Conversion, Null
    - Converting journeys:   Start -> ch1 -> ch2 -> ... -> Conversion
    - Non-converting:        Start -> ch1 -> ch2 -> ... -> Null

    Returns
    -------
    dict : {state: {next_state: probability}}
    """
    brand_journeys = df_journeys[df_journeys["Brand_ID"] == brand]

    # Count raw transitions
    transitions = defaultdict(lambda: defaultdict(int))

    for _, row in brand_journeys.iterrows():
        path = row["Path"]
        converted = row["Converted"]

        # Start -> first channel
        transitions["Start"][path[0]] += 1

        # Channel -> channel transitions
        for i in range(len(path) - 1):
            transitions[path[i]][path[i + 1]] += 1

        # Last channel -> absorbing state
        if converted:
            transitions[path[-1]]["Conversion"] += 1
        else:
            transitions[path[-1]]["Null"] += 1

    # Normalize to probabilities
    matrix = {}
    for state, targets in transitions.items():
        total = sum(targets.values())
        matrix[state] = {t: c / total for t, c in targets.items()}

    return matrix


def _compute_conversion_probability(matrix: dict) -> float:
    """
    Compute P(Conversion | Start) using absorption probability theory.

    For a Markov chain with transient states (Start + channels) and
    absorbing states (Conversion, Null):
      N = (I - Q)^(-1)   (fundamental matrix)
      B = N * R           (absorption probabilities)
      P(Conversion|Start) = B[Start, Conversion]

    Returns
    -------
    float : probability of reaching Conversion from Start
    """
    # Identify states
    all_states = set(matrix.keys())
    for targets in matrix.values():
        all_states |= set(targets.keys())

    absorbing = {"Conversion", "Null"}
    transient = sorted(all_states - absorbing)

    if not transient or "Start" not in transient:
        return 0.0

    n = len(transient)
    state_idx = {s: i for i, s in enumerate(transient)}

    # Build Q matrix (transient -> transient)
    Q = np.zeros((n, n))
    for state in transient:
        if state in matrix:
            for target, prob in matrix[state].items():
                if target in state_idx:
                    Q[state_idx[state]][state_idx[target]] = prob

    # Build R matrix (transient -> absorbing)
    absorbing_list = sorted(absorbing)
    abs_idx = {s: i for i, s in enumerate(absorbing_list)}
    R = np.zeros((n, len(absorbing_list)))
    for state in transient:
        if state in matrix:
            for target, prob in matrix[state].items():
                if target in abs_idx:
                    R[state_idx[state]][abs_idx[target]] = prob

    # Fundamental matrix N = (I - Q)^(-1)
    try:
        I = np.eye(n)
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        return 0.0

    # Absorption probabilities B = N * R
    B = N @ R

    # P(Conversion | Start)
    start_idx = state_idx.get("Start")
    conv_idx = abs_idx.get("Conversion")

    if start_idx is not None and conv_idx is not None:
        return float(B[start_idx][conv_idx])
    return 0.0


def _removal_effect(matrix: dict, channel_to_remove: str) -> float:
    """
    Compute the conversion probability when a channel is removed.

    "Removing" a channel means redirecting all transitions INTO that
    channel to Null instead (the user drops off).

    Returns
    -------
    float : P(Conversion | Start) with the channel removed
    """
    modified = {}
    for state, targets in matrix.items():
        if state == channel_to_remove:
            # Remove this state entirely
            continue
        new_targets = {}
        for target, prob in targets.items():
            if target == channel_to_remove:
                # Redirect to Null
                new_targets["Null"] = new_targets.get("Null", 0) + prob
            else:
                new_targets[target] = prob
        modified[state] = new_targets

    return _compute_conversion_probability(modified)


def markov_attribution(df_journeys: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Markov Chain attribution for each brand.

    For each brand:
    1. Build the transition matrix from all user journeys
    2. Compute baseline P(Conversion | Start)
    3. For each channel, compute removal effect
    4. Attribution = normalized removal effects

    Returns
    -------
    pd.DataFrame with columns:
        Brand_ID, Channel, Baseline_Conv_Prob, Removal_Conv_Prob,
        Removal_Effect, Markov_Attribution(%)
    """
    print("\n" + "=" * 70)
    print("  PHASE 3B: MARKOV CHAIN ATTRIBUTION")
    print("=" * 70)

    brands = sorted(df_journeys["Brand_ID"].unique())
    results = []

    for brand in brands:
        print(f"\n  Processing {brand} ...")

        matrix = _build_transition_matrix(df_journeys, brand)
        baseline_prob = _compute_conversion_probability(matrix)

        # Get all channels for this brand (exclude Start, Conversion, Null)
        channels = sorted(set(matrix.keys()) - {"Start", "Conversion", "Null"})

        brand_effects = {}
        for ch in channels:
            removed_prob = _removal_effect(matrix, ch)
            effect = baseline_prob - removed_prob
            brand_effects[ch] = max(effect, 0)  # Can't have negative attribution

        # Normalize to get percentages
        total_effect = sum(brand_effects.values())
        for ch in channels:
            attr_pct = (brand_effects[ch] / total_effect * 100
                        if total_effect > 0 else 0)
            results.append({
                "Brand_ID": brand,
                "Channel": ch,
                "Baseline_Conv_Prob": baseline_prob,
                "Removal_Conv_Prob": _removal_effect(matrix, ch),
                "Removal_Effect": brand_effects[ch],
                "Markov_Attribution(%)": attr_pct,
            })

        print(f"    Baseline P(conv) = {baseline_prob:.6f}")
        for ch in channels:
            attr_pct = (brand_effects[ch] / total_effect * 100
                        if total_effect > 0 else 0)
            print(f"    {ch:<20s} : removal effect = {brand_effects[ch]:.6f} "
                  f"-> {attr_pct:.1f}%")

    df_markov = pd.DataFrame(results)
    return df_markov


# ---------------------------------------------------------------------------
# Phase 3C: Shapley Value Attribution
# ---------------------------------------------------------------------------
def _conversion_rate_for_subset(
    df_journeys: pd.DataFrame,
    brand: str,
    channel_subset: set,
) -> float:
    """
    Compute the conversion rate when only a specific subset of
    channels is available.

    We filter journeys to those where ALL channels in the path are
    within the given subset, then compute the conversion rate.

    Parameters
    ----------
    channel_subset : set
        The set of channels that are "active"

    Returns
    -------
    float : conversion rate (0 to 1)
    """
    brand_journeys = df_journeys[df_journeys["Brand_ID"] == brand]

    if not channel_subset:
        return 0.0

    # Filter to journeys that only use channels in the subset
    mask = brand_journeys["Path"].apply(
        lambda path: all(ch in channel_subset for ch in path)
    )
    filtered = brand_journeys[mask]

    if len(filtered) == 0:
        return 0.0

    return filtered["Converted"].mean()


def shapley_attribution(df_journeys: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Shapley Value attribution for each brand.

    The Shapley value gives each channel its fair share of credit
    based on its marginal contribution across ALL possible coalitions
    of channels.

    For 5 channels, there are 2^5 = 32 subsets to evaluate per brand.

    Formula:
      phi_i = SUM over S in (N \\ {i}) of:
        [|S|! * (|N|-|S|-1)! / |N|!] * [v(S + {i}) - v(S)]

    Where v(S) = conversion rate with only channels S active.

    Returns
    -------
    pd.DataFrame with columns:
        Brand_ID, Channel, Shapley_Value, Shapley_Attribution(%)
    """
    print("\n" + "=" * 70)
    print("  PHASE 3C: SHAPLEY VALUE ATTRIBUTION")
    print("=" * 70)

    brands = sorted(df_journeys["Brand_ID"].unique())
    results = []

    for brand in brands:
        print(f"\n  Processing {brand} ...")

        brand_journeys = df_journeys[df_journeys["Brand_ID"] == brand]

        # Get all channels used by this brand
        all_channels_in_paths = set()
        for path in brand_journeys["Path"]:
            all_channels_in_paths.update(path)
        channels = sorted(all_channels_in_paths)
        n = len(channels)

        print(f"    Channels: {channels}")
        print(f"    Evaluating {2**n} subsets ...")

        # Cache conversion rates for subsets to avoid recomputation
        conv_cache = {}

        def get_conv_rate(subset_frozen):
            if subset_frozen not in conv_cache:
                conv_cache[subset_frozen] = _conversion_rate_for_subset(
                    df_journeys, brand, set(subset_frozen)
                )
            return conv_cache[subset_frozen]

        # Compute Shapley value for each channel
        shapley_values = {}
        for channel in channels:
            others = [c for c in channels if c != channel]
            phi = 0.0

            for subset_size in range(len(others) + 1):
                for subset in combinations(others, subset_size):
                    subset_set = frozenset(subset)
                    with_channel = frozenset(subset) | {channel}

                    v_with = get_conv_rate(with_channel)
                    v_without = get_conv_rate(subset_set)

                    marginal = v_with - v_without
                    weight = (factorial(len(subset_set)) *
                              factorial(n - len(subset_set) - 1)) / factorial(n)
                    phi += weight * marginal

            shapley_values[channel] = phi

        # Normalize to percentages
        total_shapley = sum(shapley_values.values())
        for ch in channels:
            attr_pct = (shapley_values[ch] / total_shapley * 100
                        if total_shapley > 0 else 0)
            results.append({
                "Brand_ID": brand,
                "Channel": ch,
                "Shapley_Value": shapley_values[ch],
                "Shapley_Attribution(%)": attr_pct,
            })
            print(f"    {ch:<20s} : shapley = {shapley_values[ch]:.6f} "
                  f"-> {attr_pct:.1f}%")

    df_shapley = pd.DataFrame(results)
    return df_shapley


# ---------------------------------------------------------------------------
# Phase 3D: Channel Role Classification
# ---------------------------------------------------------------------------
def classify_channel_roles(
    df_journeys: pd.DataFrame,
    df_markov: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify each channel's funnel role per brand based on its
    positional frequency in user journeys.

    Roles:
      - Primer/Introducer  : Appears as FIRST touch in >30% of journeys
      - Closer/Converter   : Appears as LAST touch in >30% of journeys
      - Influencer/Assist  : Everything else (mid-journey amplifier)

    Parameters
    ----------
    df_journeys : pd.DataFrame
        Journey paths.
    df_markov : pd.DataFrame
        Markov attribution results (for enrichment).

    Returns
    -------
    pd.DataFrame with columns:
        Brand_ID, Channel, First_Touch_%, Last_Touch_%, Mid_Touch_%,
        Funnel_Role, Markov_Attribution(%)
    """
    print("\n" + "=" * 70)
    print("  PHASE 3D: CHANNEL ROLE CLASSIFICATION")
    print("=" * 70)

    brands = sorted(df_journeys["Brand_ID"].unique())
    results = []

    for brand in brands:
        brand_j = df_journeys[
            (df_journeys["Brand_ID"] == brand)
            & (df_journeys["Converted"])
            & (df_journeys["Path_Length"] >= 1)
        ]

        if len(brand_j) == 0:
            continue

        total_journeys = len(brand_j)

        # Count first-touch, last-touch, and mid-touch frequencies
        first_counts = defaultdict(int)
        last_counts = defaultdict(int)
        mid_counts = defaultdict(int)
        all_channels = set()

        for _, row in brand_j.iterrows():
            path = row["Path"]
            all_channels.update(path)

            first_counts[path[0]] += 1
            last_counts[path[-1]] += 1

            for ch in path[1:-1]:  # Mid-journey channels
                mid_counts[ch] += 1

        for ch in sorted(all_channels):
            first_pct = first_counts.get(ch, 0) / total_journeys * 100
            last_pct = last_counts.get(ch, 0) / total_journeys * 100
            mid_pct = mid_counts.get(ch, 0) / total_journeys * 100

            # Classification logic
            if first_pct >= last_pct and first_pct >= mid_pct and first_pct > 25:
                role = "Primer/Introducer"
            elif last_pct >= first_pct and last_pct >= mid_pct and last_pct > 25:
                role = "Closer/Converter"
            elif mid_pct > 0:
                role = "Influencer/Assist"
            else:
                role = "Influencer/Assist"

            # Get Markov attribution for this brand-channel
            markov_row = df_markov[
                (df_markov["Brand_ID"] == brand)
                & (df_markov["Channel"] == ch)
            ]
            markov_attr = (markov_row["Markov_Attribution(%)"].values[0]
                           if len(markov_row) > 0 else 0)

            results.append({
                "Brand_ID": brand,
                "Channel": ch,
                "First_Touch(%)": round(first_pct, 1),
                "Last_Touch(%)": round(last_pct, 1),
                "Mid_Touch(%)": round(mid_pct, 1),
                "Funnel_Role": role,
                "Markov_Attribution(%)": round(markov_attr, 1),
            })

    df_roles = pd.DataFrame(results)

    # Print summary
    for brand in brands:
        brand_roles = df_roles[df_roles["Brand_ID"] == brand]
        print(f"\n  {brand}:")
        print(f"    {'Channel':<20s} {'1st%':>6} {'Last%':>6} {'Mid%':>6} "
              f"{'Markov%':>8}  {'Role'}")
        print("    " + "-" * 70)
        for _, row in brand_roles.iterrows():
            print(f"    {row['Channel']:<20s} {row['First_Touch(%)']:>5.1f}% "
                  f"{row['Last_Touch(%)']:>5.1f}% {row['Mid_Touch(%)']:>5.1f}% "
                  f"{row['Markov_Attribution(%)']:>7.1f}%  {row['Funnel_Role']}")

    return df_roles


# ---------------------------------------------------------------------------
# Phase 3E: Attribution Comparison
# ---------------------------------------------------------------------------
def build_attribution_comparison(
    df_lc_agg: pd.DataFrame,
    df_markov: pd.DataFrame,
    df_shapley: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison of all three attribution models.

    Parameters
    ----------
    df_lc_agg : pd.DataFrame
        Last-click aggregated (Brand_ID, Channel, LC_Conversions, LC_Share(%))
    df_markov : pd.DataFrame
        Markov attribution (Brand_ID, Channel, Markov_Attribution(%))
    df_shapley : pd.DataFrame
        Shapley attribution (Brand_ID, Channel, Shapley_Attribution(%))

    Returns
    -------
    pd.DataFrame with all three models side by side + delta columns
    """
    print("\n" + "=" * 70)
    print("  PHASE 3E: ATTRIBUTION MODEL COMPARISON")
    print("=" * 70)

    # Rename LC column for consistency
    lc = df_lc_agg[["Brand_ID", "Channel", "LC_Share(%)"]].copy()
    lc = lc.rename(columns={"LC_Share(%)": "LastClick(%)"})

    markov = df_markov[["Brand_ID", "Channel", "Markov_Attribution(%)"]].copy()
    markov = markov.rename(columns={"Markov_Attribution(%)": "Markov(%)"})

    shapley = df_shapley[["Brand_ID", "Channel", "Shapley_Attribution(%)"]].copy()
    shapley = shapley.rename(columns={"Shapley_Attribution(%)": "Shapley(%)"})

    # Merge all three
    comparison = lc.merge(markov, on=["Brand_ID", "Channel"], how="outer")
    comparison = comparison.merge(shapley, on=["Brand_ID", "Channel"], how="outer")
    comparison = comparison.fillna(0)

    # Delta columns: how much does multi-touch differ from last-click?
    comparison["Delta_Markov"] = comparison["Markov(%)"] - comparison["LastClick(%)"]
    comparison["Delta_Shapley"] = comparison["Shapley(%)"] - comparison["LastClick(%)"]

    comparison = comparison.sort_values(["Brand_ID", "Channel"]).reset_index(drop=True)

    # Print comparison
    for brand in sorted(comparison["Brand_ID"].unique()):
        brand_comp = comparison[comparison["Brand_ID"] == brand]
        print(f"\n  --- {brand} ---")
        print(f"    {'Channel':<20s} {'LC%':>7} {'Markov%':>8} {'Shapley%':>9} "
              f"{'D(Markov)':>10} {'D(Shapley)':>11}")
        print("    " + "-" * 68)
        for _, row in brand_comp.iterrows():
            dm = row["Delta_Markov"]
            ds = row["Delta_Shapley"]
            dm_sign = "+" if dm >= 0 else ""
            ds_sign = "+" if ds >= 0 else ""
            print(f"    {row['Channel']:<20s} {row['LastClick(%)']:>6.1f}% "
                  f"{row['Markov(%)']:>7.1f}% {row['Shapley(%)']:>8.1f}% "
                  f"{dm_sign}{dm:>9.1f}% {ds_sign}{ds:>10.1f}%")

    # Find biggest over/under-credited channels
    print("\n  BIGGEST ATTRIBUTION SHIFTS (Markov vs Last-Click):")
    top_over = comparison.nlargest(5, "Delta_Markov")
    top_under = comparison.nsmallest(5, "Delta_Markov")

    print("    Over-credited by Last-Click (Markov gives LESS credit):")
    for _, row in top_under.iterrows():
        print(f"      {row['Brand_ID']} {row['Channel']:<20s}: "
              f"LC={row['LastClick(%)']:.1f}% vs Markov={row['Markov(%)']:.1f}% "
              f"(delta: {row['Delta_Markov']:+.1f}%)")

    print("    Under-credited by Last-Click (Markov gives MORE credit):")
    for _, row in top_over.iterrows():
        print(f"      {row['Brand_ID']} {row['Channel']:<20s}: "
              f"LC={row['LastClick(%)']:.1f}% vs Markov={row['Markov(%)']:.1f}% "
              f"(delta: {row['Delta_Markov']:+.1f}%)")

    return comparison


# ---------------------------------------------------------------------------
# Master Phase 3 runner
# ---------------------------------------------------------------------------
def run_phase3(
    df_clean: pd.DataFrame,
    df_lc_agg: pd.DataFrame,
) -> dict:
    """
    Execute the complete Phase 3 pipeline.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Clean touchpoints from Phase 1.
    df_lc_agg : pd.DataFrame
        Aggregated last-click attribution from Phase 2.

    Returns
    -------
    dict with keys:
        "journeys"    : User journey paths DataFrame
        "markov"      : Markov attribution DataFrame
        "shapley"     : Shapley attribution DataFrame
        "roles"       : Channel role classification DataFrame
        "comparison"  : Three-model comparison DataFrame
    """
    print("\n" + "#" * 70)
    print("  ROI LENS -- PHASE 3: MULTI-TOUCH ATTRIBUTION ENGINE")
    print("#" * 70)

    # 3A: Build journey paths
    df_journeys = build_journey_paths(df_clean)

    # 3B: Markov Chain attribution
    df_markov = markov_attribution(df_journeys)

    # 3C: Shapley Value attribution
    df_shapley = shapley_attribution(df_journeys)

    # 3D: Channel role classification
    df_roles = classify_channel_roles(df_journeys, df_markov)

    # 3E: Attribution comparison
    df_comparison = build_attribution_comparison(
        df_lc_agg, df_markov, df_shapley
    )

    # Summary
    print("\n" + "=" * 70)
    print("  PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"  Journey paths    : {len(df_journeys):,}")
    print(f"  Markov results   : {len(df_markov)} (brand x channel)")
    print(f"  Shapley results  : {len(df_shapley)} (brand x channel)")
    print(f"  Role classif.    : {len(df_roles)} (brand x channel)")
    print(f"  Comparison rows  : {len(df_comparison)}")

    return {
        "journeys": df_journeys,
        "markov": df_markov,
        "shapley": df_shapley,
        "roles": df_roles,
        "comparison": df_comparison,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_all
    from src.data_cleaner import clean_data
    from src.funnel_analysis import run_phase2

    # Load and clean
    print("Loading data ...")
    df_tp, df_up, df_cs, df_merged = load_all()
    df_clean, bot_report = clean_data(
        df_merged, run_timestamp_check=False, verbose=False
    )

    # Run Phase 2 to get last-click baseline
    print("\nRunning Phase 2 for baseline ...")
    phase2 = run_phase2(df_clean, df_cs)

    # Run Phase 3
    phase3 = run_phase3(df_clean, phase2["attr_agg"])

    # Save outputs
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase3["markov"].to_csv(out_dir / "markov_attribution.csv", index=False)
    phase3["shapley"].to_csv(out_dir / "shapley_attribution.csv", index=False)
    phase3["roles"].to_csv(out_dir / "channel_roles.csv", index=False)
    phase3["comparison"].to_csv(out_dir / "attribution_comparison.csv", index=False)

    # Save journey stats (not full paths - too large)
    journey_stats = phase3["journeys"][
        ["User_ID", "Brand_ID", "Path_Str", "Converted", "Path_Length"]
    ]
    journey_stats.to_csv(out_dir / "journey_paths.csv", index=False)

    print(f"\n[Phase 3] All outputs saved to {out_dir}")
