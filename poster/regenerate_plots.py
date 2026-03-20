#!/usr/bin/env python3
"""
Regenerate poster figures with poster-specific titles.

- Optimal policy plots: no title/subtitle
- Observed and accuracy heatmaps: subtitle (content in parens) becomes main title, no parens
- Team performance: remove subtitle; for pp_team_performance keep "Closer to 0 = better"

Output: poster/figures/ (de-linked from results/; copies static assets from other locations)
"""

import os
import shutil
import sys

import pandas as pd

# Add code directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
code_dir = os.path.join(project_root, "code")
sys.path.insert(0, code_dir)

from data_prep import prepare_ends, load_data, build_start_of_end_df
from ep_end import (
    train_end_differential_distribution_model,
    train_early_quit_model,
)
from ep_policy import compute_pp_policy_heatmap, plot_pp_policy_heatmap
from elo import compute_elo_ratings
from eda_pp_decisions import analyze_pp_decision_states, plot_pp_decision_distributions
from evaluate_pp_decisions import (
    evaluate_pp_decisions,
    aggregate_by_team,
    plot_team_performance,
    plot_decision_patterns,
)


def main():
    poster_figures = os.path.join(project_root, "poster", "figures")
    # De-link: replace symlink with real directory so poster has its own figures
    if os.path.islink(poster_figures):
        os.unlink(poster_figures)
        print("Removed poster/figures symlink")
    os.makedirs(poster_figures, exist_ok=True)
    print(f"Output: {poster_figures}")
    print()

    data_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")

    # Load data
    print("Loading data...")
    stones, teams, games_df, ends, competitors, competition = load_data(data_dir=data_dir)
    ends_prep = prepare_ends(ends)
    end_level_df = build_start_of_end_df(ends_prep, stones, games_df)
    print(f"  Built {len(end_level_df):,} end-level rows")

    # Train models
    print("Training EP model...")
    ep_model, _, _, differential_classes, class_to_diff = train_end_differential_distribution_model(
        end_level_df, is_extra_end=False
    )

    print("Training early quit model...")
    try:
        early_quit_model, _, _ = train_early_quit_model(end_level_df)
    except Exception as e:
        print(f"  Warning: {e}")
        early_quit_model = None

    elo_ratings = compute_elo_ratings(games_df)
    elo_bucket_size = 10.0
    score_diff_clip = (-10, 10)

    # 1. Optimal policy plots (no title/subtitle)
    print("Generating optimal policy heatmaps...")
    pp_policy_opp_saved = compute_pp_policy_heatmap(
        ep_model, differential_classes, class_to_diff,
        score_range=(-5, 5), elo_diff=0.0, opp_pp_avail=1,
        elo_bucket_size=elo_bucket_size,
        score_diff_clip=score_diff_clip,
        early_quit_model=early_quit_model,
        extra_end_ep_model=None,
        extra_end_differential_classes=None,
        extra_end_class_to_diff=None,
    )
    plot_pp_policy_heatmap(
        pp_policy_opp_saved,
        save_path=os.path.join(poster_figures, "pp_heatmap_opp_saved.png"),
        for_poster=True,
    )

    pp_policy_opp_used = compute_pp_policy_heatmap(
        ep_model, differential_classes, class_to_diff,
        score_range=(-5, 5), elo_diff=0.0, opp_pp_avail=0,
        elo_bucket_size=elo_bucket_size,
        score_diff_clip=score_diff_clip,
        early_quit_model=early_quit_model,
        extra_end_ep_model=None,
        extra_end_differential_classes=None,
        extra_end_class_to_diff=None,
    )
    plot_pp_policy_heatmap(
        pp_policy_opp_used,
        save_path=os.path.join(poster_figures, "pp_heatmap_opp_used.png"),
        for_poster=True,
    )

    # 2. Observed PP usage heatmap (subtitle as main title)
    print("Generating observed PP usage heatmap...")
    decision_points = analyze_pp_decision_states(end_level_df, elo_ratings)
    plot_pp_decision_distributions(
        decision_points, poster_figures, for_poster=True
    )

    # 3. Accuracy heatmap and team performance (need evaluation)
    print("Evaluating PP decisions...")
    results_df = evaluate_pp_decisions(
        end_level_df, ep_model, differential_classes, class_to_diff,
        early_quit_model, elo_ratings,
        elo_bucket_size=elo_bucket_size,
        score_diff_clip=score_diff_clip,
    )

    teams_df = pd.read_csv(
        os.path.join(data_dir, "Teams.csv"), low_memory=False
    )
    team_names = teams_df.groupby("TeamID")["Name"].first().reset_index()
    team_stats = aggregate_by_team(results_df)
    team_stats = team_stats.merge(team_names, on="TeamID", how="left")
    team_stats = team_stats.rename(columns={"Name": "TeamName"})

    print("Generating accuracy heatmap and team performance plots...")
    plot_decision_patterns(results_df, poster_figures, for_poster=True)
    plot_team_performance(team_stats, poster_figures, for_poster=True)

    # 4. Copy static assets
    print("Copying static assets...")
    copies = [
        (os.path.join(project_root, "poster", "pp-setup.jpg"), "pp-setup.jpg"),
        (os.path.join(project_root, "research-note", "figures", "qr-app.png"), "qr-app.png"),
    ]
    for src, name in copies:
        dst = os.path.join(poster_figures, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {name}")
        else:
            print(f"  Warning: {src} not found")

    print()
    print("Done. Poster figures saved to poster/figures/")


if __name__ == "__main__":
    main()
