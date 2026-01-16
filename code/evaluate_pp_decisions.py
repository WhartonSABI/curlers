"""
Evaluate actual Power Play deployment decisions vs optimal decisions.

This script compares teams' actual PP usage decisions against the optimal
policy from the DP model, computing win probability gains/losses by team.
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
from ep_end import (
    train_end_differential_distribution_model,
    train_early_quit_model
)
from ep_policy import dp_value
from elo import compute_elo_ratings


def compute_pp_availability_at_start(end_level_df):
    """
    Compute PP availability at the start of each end.

    Note: The base pipeline uses a strict "<" check, which marks the end where PP
    is used as unavailable. For decision analysis, we need availability at the
    start of the end, so we use "<=" (available through the end it is used).
    """
    df = end_level_df.copy()
    # Identify when each ref team used PP (first end with PPUsedThisEnd == 1)
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
    # Availability at start of end: available if not used yet OR used in this end
    df["RefPPAvailableAtStart"] = (
        df["RefPPUsedEnd"].isna() | (df["EndID"] <= df["RefPPUsedEnd"])
    ).astype(int)
    return df


def evaluate_pp_decisions(
    end_level_df,
    ep_model,
    differential_classes,
    class_to_diff,
    early_quit_model,
    elo_ratings,
    elo_bucket_size=10.0,
    score_diff_clip=(-10, 10)
):
    """
    Evaluate actual PP decisions vs optimal decisions.
    
    Parameters
    ----------
    end_level_df : pd.DataFrame
        End-level dataframe with actual PP decisions
    ep_model : XGBClassifier
        Trained EP model
    differential_classes : np.ndarray
        Array of possible differential values
    class_to_diff : dict
        Mapping from class index to differential value
    early_quit_model : XGBClassifier
        Early quit model
    elo_ratings : dict
        Dictionary mapping TeamID to Elo rating
    elo_bucket_size : float
        Elo bucket size for DP
    score_diff_clip : tuple
        Score diff clipping range
    
    Returns
    -------
    pd.DataFrame
        Evaluation results with columns:
        - TeamID, EndID, ScoreDiff, Hammer, RefPPAvail, OppPPAvail
    - ActualDecision (1 if used PP, 0 if not)
    - OptimalDecision (1 if should use PP, 0 if not)
    - ActualWP (win prob with actual decision)
    - OptimalWP (win prob with optimal decision)
    - WPDiff (actual - optimal, should be ≤ 0)
    - DecisionCorrect (1 if actual == optimal, 0 otherwise)
    """
    # Compute PP availability at start of end (<= logic)
    end_level_df = compute_pp_availability_at_start(end_level_df)

    # Filter to PP decision points: ref has hammer and PP available at start
    decision_points = end_level_df[
        (end_level_df["RefHasHammerStartOfEnd"] == 1) &
        (end_level_df["RefPPAvailableAtStart"] == 1) &
        (end_level_df["EndID"].between(1, 8))
    ].copy()
    
    print(f"Evaluating {len(decision_points):,} PP decision points...")
    
    results = []
    value_cache = {}  # Shared cache for efficiency
    
    for idx, row in decision_points.iterrows():
        # Get game state
        end = int(row["EndID"])
        score_diff = int(row["RefScoreDiffStartOfEnd"])
        hammer = 1  # Ref has hammer (by filter)
        ref_pp_avail = 1  # Ref has PP (by filter)
        opp_pp_avail = int(row["OppPPAvailableBeforeEnd"])
        
        # Get Elo difference
        ref_team = int(row["RefTeamID"])
        opp_team = int(row["OppTeamID"])
        ref_elo = elo_ratings.get(ref_team, 1500.0)
        opp_elo = elo_ratings.get(opp_team, 1500.0)
        elo_diff = ref_elo - opp_elo
        
        # Actual decision: did ref team use PP this end?
        actual_decision = 1 if row["PPUsedThisEnd"] == 1 else 0
        
        # Compute win probability for both possible actions
        wp_use_pp = dp_value(
            end, score_diff, hammer, 0, opp_pp_avail,  # Use PP (ref_pp_avail = 0 after using)
            ep_model, differential_classes, class_to_diff, value_cache,
            elo_diff, elo_bucket_size, score_diff_clip,
            early_quit_model, None, None, None
        )
        wp_save_pp = dp_value(
            end, score_diff, hammer, 1, opp_pp_avail,  # Save PP (ref_pp_avail = 1 still available)
            ep_model, differential_classes, class_to_diff, value_cache,
            elo_diff, elo_bucket_size, score_diff_clip,
            early_quit_model, None, None, None
        )
        
        # Optimal decision is the one with higher win probability
        optimal_decision = 1 if wp_use_pp > wp_save_pp else 0
        optimal_wp = max(wp_use_pp, wp_save_pp)
        
        # Actual win probability is the one corresponding to their actual decision
        actual_wp = wp_use_pp if actual_decision == 1 else wp_save_pp
        
        # Win probability difference: actual - optimal (should be ≤ 0)
        # Negative = lost WP by suboptimal decision, 0 = made optimal decision
        wp_diff = actual_wp - optimal_wp
        
        # Sanity check: wp_diff should never be positive (allowing small numerical tolerance)
        if wp_diff > 1e-6:
            print(f"Warning: Unexpected positive wp_diff: {wp_diff:.6f} for Team {ref_team}, End {end}")
            wp_diff = 0.0
        
        results.append({
            "TeamID": ref_team,
            "EndID": end,
            "ScoreDiff": score_diff,
            "Hammer": hammer,
            "RefPPAvail": ref_pp_avail,
            "OppPPAvail": opp_pp_avail,
            "EloDiff": elo_diff,
            "ActualDecision": actual_decision,
            "OptimalDecision": optimal_decision,
            "ActualWP": actual_wp,
            "OptimalWP": optimal_wp,
            "WPDiff": wp_diff,  # actual - optimal (≤ 0)
            "DecisionCorrect": 1 if actual_decision == optimal_decision else 0
        })
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1:,} / {len(decision_points):,} decisions...")
    
    return pd.DataFrame(results)


def aggregate_by_team(results_df):
    """
    Aggregate PP decision evaluation results by team.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from evaluate_pp_decisions()
    
    Returns
    -------
    pd.DataFrame
        Aggregated results by team with columns:
        - TeamID
        - NumDecisions
        - NumCorrect
        - Accuracy (fraction of optimal decisions)
        - TotalWPDiff (sum of WP differences, actual - optimal)
        - AvgWPDiff (average WP difference per decision)
        - TotalWPLost (sum of negative WP differences)
        - MaxWPLost (largest single decision WP loss)
    """
    team_stats = results_df.groupby("TeamID").agg({
        "DecisionCorrect": ["count", "sum"],
        "WPDiff": ["sum", "mean", lambda x: x[x < 0].sum(), "min"]
    }).reset_index()
    
    team_stats.columns = [
        "TeamID",
        "NumDecisions",
        "NumCorrect",
        "TotalWPDiff",
        "AvgWPDiff",
        "TotalWPLost",
        "MaxWPLost"
    ]
    
    team_stats["Accuracy"] = team_stats["NumCorrect"] / team_stats["NumDecisions"]
    team_stats["TotalWPLost"] = team_stats["TotalWPLost"].fillna(0)
    team_stats["MaxWPLost"] = team_stats["MaxWPLost"].fillna(0)
    
    # Sort by total WP difference (best to worst, i.e., least negative to most negative)
    team_stats = team_stats.sort_values("TotalWPDiff", ascending=False)
    
    return team_stats


def plot_team_performance(team_stats, save_dir):
    """Plot team-level PP decision performance."""
    
    # Filter to teams with at least 5 decisions
    team_stats_filtered = team_stats[team_stats["NumDecisions"] >= 5].copy()
    
    # Order teams by total WP difference (most negative to least negative)
    ordered_teams = team_stats_filtered.sort_values("TotalWPDiff", ascending=True).reset_index(drop=True)
    ordered_by_accuracy = team_stats_filtered.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    # Plot 1: Total WP difference (actual - optimal) by team
    plt.figure(figsize=(14, 10))
    top_teams = ordered_teams
    # Color: closer to 0 (optimal) is better, more negative is worse
    perf_min = top_teams["TotalWPDiff"].min()
    perf_max = top_teams["TotalWPDiff"].max()
    perf_norm = (top_teams["TotalWPDiff"] - perf_min) / (perf_max - perf_min + 1e-9)
    colors = plt.cm.RdYlGn(perf_norm)
    plt.barh(range(len(top_teams)), top_teams["TotalWPDiff"], color=colors, alpha=0.7)
    labels = top_teams["TeamName"].fillna(top_teams["TeamID"].astype(str)).tolist()
    plt.yticks(range(len(top_teams)), labels)
    plt.xlabel("Total Win Probability Difference (Actual - Optimal)")
    plt.title("PP Decision Performance by Team\n(All teams ranked by total WP difference, ≥5 decisions)\nCloser to 0 = better (optimal decisions)")
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Optimal (0)')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_team_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Decision accuracy by team (sorted by accuracy)
    plt.figure(figsize=(14, 10))
    top_teams = ordered_by_accuracy
    acc_min = top_teams["Accuracy"].min()
    acc_max = top_teams["Accuracy"].max()
    acc_norm = (top_teams["Accuracy"] - acc_min) / (acc_max - acc_min + 1e-9)
    colors = plt.cm.RdYlGn(acc_norm)
    plt.barh(range(len(top_teams)), top_teams["Accuracy"], color=colors, alpha=0.7)
    labels = top_teams["TeamName"].fillna(top_teams["TeamID"].astype(str)).tolist()
    plt.yticks(range(len(top_teams)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Decision Accuracy (Fraction of Optimal Decisions)")
    plt.title("PP Decision Accuracy by Team\n(All teams ranked by total WP difference, ≥5 decisions)")
    plt.xlim(0.6, 0.8)
    plt.axvline(x=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (1.0)')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_team_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_decision_patterns(results_df, save_dir):
    """Plot patterns in PP decision making."""
    
    # Plot 1: Average WP difference (actual - optimal) by end
    plt.figure(figsize=(10, 6))
    end_stats = results_df.groupby("EndID")["WPDiff"].agg(["mean", "std", "count"]).reset_index()
    end_stats = end_stats[end_stats["count"] >= 10]  # Only ends with enough data
    plt.errorbar(end_stats["EndID"], end_stats["mean"], 
                yerr=end_stats["std"], fmt='o-', capsize=5, capthick=2, color='steelblue')
    plt.xlabel("End Number")
    plt.ylabel("Average Win Probability Difference (Actual - Optimal)")
    plt.title("PP Decision Quality by End\n(Closer to 0 = better, negative = lost WP)")
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Optimal (0)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_decision_by_end.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Average WP difference by score differential
    plt.figure(figsize=(12, 6))
    score_stats = results_df.groupby("ScoreDiff")["WPDiff"].agg(["mean", "std", "count"]).reset_index()
    score_stats = score_stats[score_stats["count"] >= 5]  # Only score diffs with enough data
    score_mask = (score_stats["ScoreDiff"] >= -5) & (score_stats["ScoreDiff"] <= 5)
    score_stats = score_stats[score_mask]
    plt.errorbar(score_stats["ScoreDiff"], score_stats["mean"],
                yerr=score_stats["std"], fmt='o-', capsize=5, capthick=2, color='steelblue')
    plt.xlabel("Score Differential (Ref - Opp)")
    plt.ylabel("Average Win Probability Difference (Actual - Optimal)")
    plt.title("PP Decision Quality by Score Differential\n(Closer to 0 = better, negative = lost WP)")
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Optimal (0)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_decision_by_score.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: WP difference heatmap (End vs ScoreDiff) - shows where teams lose most WP
    plt.figure(figsize=(12, 8))
    heatmap_data = results_df.pivot_table(
        values="WPDiff", index="ScoreDiff", columns="EndID", aggfunc="mean"
    )
    # Filter to reasonable ranges
    score_mask = (heatmap_data.index >= -5) & (heatmap_data.index <= 5)
    end_mask = (heatmap_data.columns >= 1) & (heatmap_data.columns <= 8)
    heatmap_data = heatmap_data.loc[score_mask, end_mask]
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="Reds_r", vmin=-0.15, vmax=0,
                cbar_kws={'label': 'Avg WP Difference\n(Actual - Optimal, ≤ 0)'})
    plt.xlabel("End Number")
    plt.ylabel("Score Differential")
    plt.title("PP Decision Quality Heatmap\n(Average WP Difference: Actual - Optimal)\nDarker red = more WP lost")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_decision_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Decision accuracy heatmap (End vs ScoreDiff) - shows where teams make optimal decisions
    plt.figure(figsize=(12, 8))
    heatmap_data = results_df.pivot_table(
        values="DecisionCorrect", index="ScoreDiff", columns="EndID", aggfunc="mean"
    )
    count_data = results_df.pivot_table(
        values="DecisionCorrect", index="ScoreDiff", columns="EndID", aggfunc="count"
    )
    # Filter to reasonable ranges
    score_mask = (heatmap_data.index >= -5) & (heatmap_data.index <= 5)
    end_mask = (heatmap_data.columns >= 1) & (heatmap_data.columns <= 8)
    heatmap_data = heatmap_data.loc[score_mask, end_mask]
    count_data = count_data.loc[score_mask, end_mask]

    # Build annotations with accuracy and count per cell
    annot = heatmap_data.copy().astype(object)
    for i in range(annot.shape[0]):
        for j in range(annot.shape[1]):
            val = heatmap_data.iloc[i, j]
            cnt = count_data.iloc[i, j]
            if pd.isna(val) or pd.isna(cnt):
                annot.iloc[i, j] = ""
            else:
                annot.iloc[i, j] = f"{val:.2f}\n(n={int(cnt)})"

    sns.heatmap(heatmap_data, annot=annot, fmt="", cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={'label': 'Optimal Decision Rate\n(1.0 = always optimal)'})
    plt.xlabel("End Number")
    plt.ylabel("Score Differential")
    plt.title("PP Decision Accuracy Heatmap\n(Fraction of Optimal Decisions by Situation)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pp_accuracy_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation pipeline."""
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "raw")
    results_dir = os.path.join(project_root, "results", "power-play")
    processed_dir = os.path.join(project_root, "data", "processed")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("=" * 80)
    print("PP Decision Evaluation Pipeline")
    print("=" * 80)
    print()
    
    # Load and prepare data
    print("Step 1: Loading and preparing data...")
    stones, teams, games_df, ends, competitors, competition = load_data(data_dir=data_dir)
    ends_prep = prepare_ends(ends)
    games = games_df
    print(f"  - Loaded {len(games):,} games")
    
    print("  Building end-level dataframe...")
    end_level_df = build_start_of_end_df(ends_prep, stones, games)
    print(f"  Built {len(end_level_df):,} end-level rows")
    
    # Train models (same as main.py)
    print("Step 2: Training models...")
    print("  Training EP model...")
    ep_model, ep_train_df, ep_val_df, differential_classes, class_to_diff = train_end_differential_distribution_model(
        end_level_df, is_extra_end=False
    )
    
    print("  Training early quit model...")
    try:
        early_quit_model, _, _ = train_early_quit_model(end_level_df)
        print("  Early quit model trained")
    except Exception as e:
        print(f"  Warning: Could not train early quit model: {e}")
        early_quit_model = None
    
    # Compute Elo ratings
    print("  Computing Elo ratings...")
    elo_ratings = compute_elo_ratings(games)
    print(f"  Computed Elo ratings for {len(elo_ratings):,} teams")
    
    # Evaluate PP decisions
    print("Step 3: Evaluating PP decisions...")
    results_df = evaluate_pp_decisions(
        end_level_df, ep_model, differential_classes, class_to_diff,
        early_quit_model, elo_ratings
    )
    print(f"  Evaluated {len(results_df):,} PP decision points")
    
    # Aggregate by team
    print("Step 4: Aggregating by team...")
    team_stats = aggregate_by_team(results_df)
    teams_df = pd.read_csv(os.path.join(data_dir, "Teams.csv"), low_memory=False)
    team_names = teams_df.groupby("TeamID")["Name"].first().reset_index()
    team_stats = team_stats.merge(team_names, on="TeamID", how="left")
    team_stats = team_stats.rename(columns={"Name": "TeamName"})
    print(f"  Aggregated results for {len(team_stats):,} teams")
    
    # Save results
    print("Step 5: Saving results...")
    results_df.to_csv(os.path.join(processed_dir, "pp_decision_evaluation.csv"), index=False)
    team_stats.to_csv(os.path.join(processed_dir, "pp_team_stats.csv"), index=False)
    print("  Saved detailed results and team statistics to data/processed")
    
    # Generate plots
    print("Step 6: Generating plots...")
    plot_team_performance(team_stats, results_dir)
    plot_decision_patterns(results_df, results_dir)
    print("  Generated all plots")
    
    # Print summary statistics
    print()
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total PP decisions evaluated: {len(results_df):,}")
    print(f"Optimal decision rate: {results_df['DecisionCorrect'].mean():.2%}")
    print(f"Average WP difference (actual - optimal): {results_df['WPDiff'].mean():.4f}")
    print(f"  (Should be ≤ 0, negative = lost WP by suboptimal decisions)")
    print(f"Total WP lost (suboptimal decisions): {results_df[results_df['WPDiff'] < 0]['WPDiff'].sum():.4f}")
    print(f"Largest single decision WP loss: {results_df['WPDiff'].min():.4f}")
    print()
    print("Top 5 teams (closest to optimal):")
    for idx, row in team_stats.head(5).iterrows():
        print(f"  Team {int(row['TeamID'])}: {row['TotalWPDiff']:.4f} WP diff ({row['NumDecisions']} decisions, {row['Accuracy']:.2%} accuracy)")
    print()
    print("Bottom 5 teams (furthest from optimal):")
    for idx, row in team_stats.tail(5).iterrows():
        print(f"  Team {int(row['TeamID'])}: {row['TotalWPDiff']:.4f} WP diff ({row['NumDecisions']} decisions, {row['Accuracy']:.2%} accuracy)")
    print()
    print("=" * 80)
    print("Evaluation completed!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

