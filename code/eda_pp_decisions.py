"""
EDA: Explore game states where Power Play decisions are made.

In mixed doubles curling:
- Power Play can only be used by the team with the hammer (last stone advantage)
- Each team can use Power Play only once per game
- The decision is made at the start of an end, before any stones are thrown
- Power Play moves the positioned stones (placed before the end) to the left or right side

This script visualizes the game states (end, score diff, hammer, PP availability)
where teams actually made PP decisions in our dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_prep import prepare_ends, load_data, build_start_of_end_df
from elo import compute_elo_ratings


def compute_pp_availability_at_start(end_level_df):
    """
    Compute PP availability at the start of each end.

    Uses "<=" so the end where PP is used is still considered available at start.
    """
    df = end_level_df.copy()
    df["RefGameTeamID"] = (
        df["CompetitionID"].astype(str) + "_" +
        df["SessionID"].astype(str) + "_" +
        df["GameID"].astype(str) + "_" +
        df["RefTeamID"].astype(str)
    )
    ref_pp_used_end = (
        df[df["PPUsedThisEnd"] == 1]
        .groupby("RefGameTeamID")["EndID"]
        .min()
        .rename("RefPPUsedEnd")
    )
    df = df.merge(ref_pp_used_end, on="RefGameTeamID", how="left")
    df["RefPPAvailableAtStart"] = (
        df["RefPPUsedEnd"].isna() | (df["EndID"] <= df["RefPPUsedEnd"])
    ).astype(int)
    return df


def analyze_pp_decision_states(end_level_df, elo_ratings):
    """
    Analyze game states where PP decisions are made.
    
    Parameters
    ----------
    end_level_df : pd.DataFrame
        End-level dataframe with PP decision information
    elo_ratings : dict
        Dictionary mapping TeamID to Elo rating
    
    Returns
    -------
    pd.DataFrame
        DataFrame with PP decision states and statistics
    """
    # Compute PP availability at start of end (<= logic)
    end_level_df = compute_pp_availability_at_start(end_level_df)

    # Filter to PP decision points: team has hammer and PP available at start
    decision_points = end_level_df[
        (end_level_df["RefHasHammerStartOfEnd"] == 1) &
        (end_level_df["RefPPAvailableAtStart"] == 1) &
        (end_level_df["EndID"].between(1, 8))
    ].copy()
    
    # Add Elo difference
    decision_points["RefTeamElo"] = decision_points["RefTeamID"].map(elo_ratings).fillna(1500.0)
    decision_points["OppTeamElo"] = decision_points["OppTeamID"].map(elo_ratings).fillna(1500.0)
    decision_points["RefEloDiff"] = decision_points["RefTeamElo"] - decision_points["OppTeamElo"]
    
    # Add decision made
    decision_points["UsedPP"] = (decision_points["PPUsedThisEnd"] == 1).astype(int)
    
    # Also check: if PPUsedThisEnd == 1, then they definitely used it (even if RefPPAvailableBeforeEnd shows 0)
    # The issue is RefPPAvailableBeforeEnd uses strict <, so if used in end N, it shows 0 for end N
    # But we want to see decision points where PP was available BEFORE they decided
    # So we should look at ends where RefPPAvailableBeforeEnd == 1 OR PPUsedThisEnd == 1
    # Actually, let's keep it simple: decision points are where they COULD use PP
    # (have hammer and PP available at start), and UsedPP shows if they DID use it
    
    # Add ends remaining
    decision_points["EndsRemaining"] = 8 - decision_points["EndID"]
    
    return decision_points


def plot_pp_decision_distributions(decision_points, save_dir):
    """Plot distributions of game states where PP decisions are made."""
    
    # Plot 1: Distribution by end
    plt.figure(figsize=(12, 6))
    end_counts = decision_points.groupby("EndID").size()
    end_pp_used = decision_points[decision_points["UsedPP"] == 1].groupby("EndID").size()
    end_pp_saved = decision_points[decision_points["UsedPP"] == 0].groupby("EndID").size()
    
    x = np.arange(1, 9)
    width = 0.35
    
    plt.bar(x - width/2, [end_pp_used.get(i, 0) for i in x], width, 
            label='Used PP', color='steelblue', alpha=0.8)
    plt.bar(x + width/2, [end_pp_saved.get(i, 0) for i in x], width,
            label='Saved PP', color='lightcoral', alpha=0.8)
    
    plt.xlabel("End Number")
    plt.ylabel("Number of PP Decision Points")
    plt.title("PP Decision Points by End\n(Team has hammer and PP available)")
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_decisions_by_end.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Distribution by score differential
    plt.figure(figsize=(14, 6))
    score_bins = np.arange(-6, 7)
    score_counts, _ = np.histogram(decision_points["RefScoreDiffStartOfEnd"], bins=score_bins)
    score_pp_used, _ = np.histogram(
        decision_points[decision_points["UsedPP"] == 1]["RefScoreDiffStartOfEnd"], 
        bins=score_bins
    )
    score_pp_saved, _ = np.histogram(
        decision_points[decision_points["UsedPP"] == 0]["RefScoreDiffStartOfEnd"],
        bins=score_bins
    )
    
    x = score_bins[:-1]
    width = 0.35
    
    plt.bar(x - width/2, score_pp_used, width, label='Used PP', color='steelblue', alpha=0.8)
    plt.bar(x + width/2, score_pp_saved, width, label='Saved PP', color='lightcoral', alpha=0.8)
    
    plt.xlabel("Score Differential (Ref Team - Opponent)")
    plt.ylabel("Number of PP Decision Points")
    plt.title("PP Decision Points by Score Differential\n(Team has hammer and PP available)")
    plt.xticks(x)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_decisions_by_score.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: PP usage rate by end
    plt.figure(figsize=(10, 6))
    end_usage = decision_points.groupby("EndID")["UsedPP"].agg(["mean", "count"]).reset_index()
    end_usage = end_usage[end_usage["count"] >= 5]  # Only ends with enough data
    
    plt.bar(end_usage["EndID"], end_usage["mean"], color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel("End Number")
    plt.ylabel("PP Usage Rate")
    plt.title("PP Usage Rate by End\n(Fraction of teams that used PP when they had the option)")
    plt.ylim(0, 1)
    plt.xticks(end_usage["EndID"])
    plt.grid(axis='y', alpha=0.3)
    for idx, row in end_usage.iterrows():
        plt.text(row["EndID"], row["mean"] + 0.02, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_usage_rate_by_end.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: PP usage rate by score differential
    plt.figure(figsize=(12, 6))
    score_usage = decision_points.groupby("RefScoreDiffStartOfEnd")["UsedPP"].agg(["mean", "count"]).reset_index()
    score_usage = score_usage[
        (score_usage["count"] >= 5) & 
        (score_usage["RefScoreDiffStartOfEnd"].between(-5, 5))
    ]
    
    plt.bar(score_usage["RefScoreDiffStartOfEnd"], score_usage["mean"], 
            color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel("Score Differential (Ref Team - Opponent)")
    plt.ylabel("PP Usage Rate")
    plt.title("PP Usage Rate by Score Differential\n(Fraction of teams that used PP when they had the option)")
    plt.ylim(0, 1)
    plt.xticks(score_usage["RefScoreDiffStartOfEnd"])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_usage_rate_by_score.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Heatmap: PP usage rate by End vs ScoreDiff
    plt.figure(figsize=(12, 8))
    heatmap_data = decision_points.pivot_table(
        values="UsedPP", index="RefScoreDiffStartOfEnd", columns="EndID", 
        aggfunc="mean"
    )
    # Filter to reasonable ranges
    score_mask = (heatmap_data.index >= -5) & (heatmap_data.index <= 5)
    end_mask = (heatmap_data.columns >= 1) & (heatmap_data.columns <= 8)
    heatmap_data = heatmap_data.loc[score_mask, end_mask]
    
    # Only show cells with at least 3 observations
    count_data = decision_points.pivot_table(
        values="UsedPP", index="RefScoreDiffStartOfEnd", columns="EndID",
        aggfunc="count"
    )
    count_data = count_data.loc[score_mask, end_mask]
    heatmap_data = heatmap_data.where(count_data >= 3)
    
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                cbar_kws={'label': 'PP Usage Rate'}, mask=heatmap_data.isna())
    plt.xlabel("End Number")
    plt.ylabel("Score Differential (Ref - Opp)")
    plt.title("PP Usage Rate Heatmap\n(Fraction of teams using PP by game state)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_usage_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Distribution of Elo differences at decision points
    plt.figure(figsize=(10, 6))
    plt.hist(decision_points["RefEloDiff"], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel("Elo Difference (Ref Team - Opponent)")
    plt.ylabel("Number of PP Decision Points")
    plt.title("Distribution of Elo Differences at PP Decision Points")
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Equal Elo')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_decisions_elo_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 7: Opponent PP availability at decision points
    plt.figure(figsize=(10, 6))
    opp_pp_counts = decision_points.groupby("OppPPAvailableBeforeEnd").size()
    labels = ['Opp PP Used', 'Opp PP Available']
    colors = ['lightcoral', 'lightgreen']
    plt.pie(opp_pp_counts.values, labels=labels, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title("Opponent PP Status at PP Decision Points\n(When ref team has hammer and PP available)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_decisions_opp_pp_status.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 8: PP usage rate by opponent PP availability
    plt.figure(figsize=(8, 6))
    opp_pp_usage = decision_points.groupby("OppPPAvailableBeforeEnd")["UsedPP"].agg(["mean", "count"]).reset_index()
    opp_pp_usage["OppPPStatus"] = opp_pp_usage["OppPPAvailableBeforeEnd"].map({0: "Opp PP Used", 1: "Opp PP Available"})
    
    plt.bar(opp_pp_usage["OppPPStatus"], opp_pp_usage["mean"], 
            color=['lightcoral', 'lightgreen'], alpha=0.7, edgecolor='black')
    plt.ylabel("PP Usage Rate")
    plt.title("PP Usage Rate by Opponent PP Status\n(When ref team has hammer and PP available)")
    plt.ylim(0, 1)
    for idx, row in opp_pp_usage.iterrows():
        plt.text(idx, row["mean"] + 0.02, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_usage_by_opp_pp_status.png"), dpi=300, bbox_inches='tight')
    plt.close()


def print_summary_statistics(decision_points):
    """Print summary statistics about PP decision points."""
    print("=" * 80)
    print("PP Decision Points Summary Statistics")
    print("=" * 80)
    print(f"Total PP decision points: {len(decision_points):,}")
    print(f"  (Team has hammer and PP available)")
    print()
    print(f"PP usage:")
    print(f"  Used PP: {decision_points['UsedPP'].sum():,} ({decision_points['UsedPP'].mean():.2%})")
    print(f"  Saved PP: {(1 - decision_points['UsedPP']).sum():,} ({(1 - decision_points['UsedPP']).mean():.2%})")
    print()
    print(f"By end:")
    for end in sorted(decision_points["EndID"].unique()):
        end_data = decision_points[decision_points["EndID"] == end]
        print(f"  End {end}: {len(end_data):,} decisions, {end_data['UsedPP'].mean():.2%} used PP")
    print()
    print(f"By score differential:")
    for score_diff in sorted(decision_points["RefScoreDiffStartOfEnd"].unique()):
        if abs(score_diff) <= 5:
            score_data = decision_points[decision_points["RefScoreDiffStartOfEnd"] == score_diff]
            if len(score_data) >= 5:
                print(f"  Score diff {score_diff:+d}: {len(score_data):,} decisions, {score_data['UsedPP'].mean():.2%} used PP")
    print()
    print(f"By opponent PP status:")
    for opp_pp in [0, 1]:
        opp_data = decision_points[decision_points["OppPPAvailableBeforeEnd"] == opp_pp]
        status = "Opp PP Used" if opp_pp == 0 else "Opp PP Available"
        print(f"  {status}: {len(opp_data):,} decisions, {opp_data['UsedPP'].mean():.2%} used PP")
    print("=" * 80)


def main():
    """Main EDA pipeline."""
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "raw")
    results_dir = os.path.join(project_root, "results", "eda")
    processed_dir = os.path.join(project_root, "data", "processed")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("=" * 80)
    print("PP Decision States EDA")
    print("=" * 80)
    print()
    print("Power Play Rules in Mixed Doubles Curling:")
    print("  - Can only be used by the team with the hammer (last stone advantage)")
    print("  - Each team can use Power Play only once per game")
    print("  - Decision is made at the start of an end, before any stones are thrown")
    print("  - Power Play moves positioned stones to left or right side")
    print()
    
    # Load and prepare data
    print("Loading and preparing data...")
    stones, teams, games_df, ends, competitors, competition = load_data(data_dir=data_dir)
    ends_prep = prepare_ends(ends)
    games = games_df
    
    print("Building end-level dataframe...")
    end_level_df = build_start_of_end_df(ends_prep, stones, games)
    print(f"Built {len(end_level_df):,} end-level rows")
    
    # Compute Elo ratings
    print("Computing Elo ratings...")
    elo_ratings = compute_elo_ratings(games)
    print(f"Computed Elo ratings for {len(elo_ratings):,} teams")
    
    # Analyze PP decision states
    print("Analyzing PP decision states...")
    decision_points = analyze_pp_decision_states(end_level_df, elo_ratings)
    print(f"Found {len(decision_points):,} PP decision points")
    
    # Print summary statistics
    print_summary_statistics(decision_points)
    
    # Generate plots
    print("Generating plots...")
    plot_pp_decision_distributions(decision_points, results_dir)
    print("Generated all plots")
    
    # Save decision points data
    output_cols = [
        "CompetitionID", "SessionID", "GameID", "EndID",
        "RefTeamID", "OppTeamID",
        "RefScoreDiffStartOfEnd",
        "RefHasHammerStartOfEnd",
        "RefPPAvailableAtStart",
        "OppPPAvailableBeforeEnd",
        "UsedPP",
        "RefEloDiff",
        "EndsRemaining"
    ]
    decision_points[output_cols].to_csv(
        os.path.join(processed_dir, "pp_decision_points.csv"), index=False
    )
    print(f"Saved decision points to: {os.path.join(processed_dir, 'pp_decision_points.csv')}")
    
    print()
    print("=" * 80)
    print("EDA completed!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
