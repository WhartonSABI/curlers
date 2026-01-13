"""
Main script to run the complete curling win probability pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_prep import prepare_ends, load_data, build_start_of_end_df
from ep_end import (
    prepare_end_level_features,
    train_end_differential_distribution_model,
    train_early_quit_model
)
from ep_policy import (
    compute_pp_policy_heatmap,
    plot_pp_policy_heatmap,
    compute_pp_delta_ep,
    test_elo_bucket_sizes
)
from decisions import aggregate_decisions

from results import (
    evaluate_ep_model,
    plot_ep_confusion_matrix,
    plot_ep_prediction_distribution,
    plot_ep_feature_importance
)

def main():
    """Main pipeline execution."""
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    results_dir = os.path.join(project_root, "results", "power-play")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("Power Play Decision Analysis Pipeline")
    print("=" * 80)
    print()
    
    # Step 1: Data Preparation
    print("Step 1: Loading and preparing data...")
    stones, teams, games_df, ends, competitors, competition = load_data(data_dir=data_dir)
    ends_prep = prepare_ends(ends)
    games = games_df  # Use games_df as games
    print(f"  - Loaded {len(stones):,} stones")
    print(f"  - Loaded {len(games):,} games")
    print()
    
    # Power play decision analysis
    print("  - Power play decision analysis...")
    print("    Building end-level start-of-end dataframe...")
    end_level_df = build_start_of_end_df(ends_prep, stones, games)
    print(f"    Built {len(end_level_df):,} end-level rows")
    
    print("    Training end-level EP distribution model (regular ends)...")
    ep_model, ep_train_df, ep_val_df, differential_classes, class_to_diff = train_end_differential_distribution_model(
        end_level_df, is_extra_end=False
    )
    print(f"    EP model trained on {len(ep_train_df):,} samples")
    print(f"    Predicting distribution over {len(differential_classes)} differential values: {differential_classes.min()} to {differential_classes.max()}")
    
    # Extra ends: use regular EP model with EndsRemaining=0 (no separate model needed)
    # Just process one extra end at a time until someone scores
    
    # Train early quit model
    print("    Training early quit probability model...")
    try:
        early_quit_model, early_quit_train_df, early_quit_val_df = train_early_quit_model(end_level_df)
        print(f"    Early quit model trained on {len(early_quit_train_df):,} samples")
        print(f"    Validation set: {len(early_quit_val_df):,} samples")
        # Evaluate early quit model
        from sklearn.metrics import roc_auc_score, log_loss
        X_early_quit_val = early_quit_val_df[["EndID", "RefHasHammerStartOfEnd", "RefScoreDiffStartOfEnd", 
                                               "RefPPAvailableBeforeEnd", "OppPPAvailableBeforeEnd", "RefEloDiff"]]
        y_early_quit_val = early_quit_val_df["EarlyQuit"]
        y_pred_proba = early_quit_model.predict_proba(X_early_quit_val)[:, 1]
        if len(np.unique(y_early_quit_val)) > 1:
            auc = roc_auc_score(y_early_quit_val, y_pred_proba)
            logloss = log_loss(y_early_quit_val, y_pred_proba)
            print(f"      AUC: {auc:.3f}, Log loss: {logloss:.4f}")
    except Exception as e:
        print(f"    Warning: Could not train early quit model: {e}")
        early_quit_model = None
    
    # EP Model Evaluation
    print("    Evaluating EP model...")
    X_ep_val = prepare_end_level_features(ep_val_df)
    y_ep_val = ep_val_df["ClassLabel"]
    ep_metrics = evaluate_ep_model(ep_model, X_ep_val, y_ep_val, differential_classes, class_to_diff)
    print(f"      Accuracy: {ep_metrics['accuracy']:.3f}")
    print(f"      Log loss: {ep_metrics['logloss']:.4f}")
    print(f"      MAE (expected): {ep_metrics['mae_expected']:.3f}")
    
    # EP Model Plots
    print("    Generating EP model plots...")
    plot_ep_confusion_matrix(
        ep_model, X_ep_val, y_ep_val, differential_classes, class_to_diff,
        save_path=os.path.join(results_dir, "ep_confusion_matrix.png")
    )
    plot_ep_prediction_distribution(
        ep_model, X_ep_val, y_ep_val, differential_classes, class_to_diff,
        save_path=os.path.join(results_dir, "ep_distribution.png")
    )
    ep_feature_importance_df = plot_ep_feature_importance(
        ep_model,
        save_path=os.path.join(results_dir, "ep_feature_importance.png")
    )
    print(f"      Top 5 EP features:")
    for idx, row in ep_feature_importance_df.head(5).iterrows():
        print(f"        {row['feature']}: {row['importance']:.4f}")
    
    # Test Elo bucket sizes
    print("    Testing Elo bucket sizes...")
    elo_bucket_results = test_elo_bucket_sizes(
        end_level_df, ep_model, differential_classes, class_to_diff,
        bucket_sizes=[10.0, 25.0, 50.0, 100.0, 200.0],
        early_quit_model=early_quit_model,
        extra_end_ep_model=None,  # No separate extra end model
        extra_end_differential_classes=None,
        extra_end_class_to_diff=None
    )
    print("    Elo bucket test results:")
    print(elo_bucket_results.to_string(index=False))
    # Choose bucket size with best cache hit rate and low value diff
    best_bucket = elo_bucket_results.loc[
        (elo_bucket_results["avg_value_diff"] < 0.001) & 
        (elo_bucket_results["cache_hit_rate"] == elo_bucket_results["cache_hit_rate"].max()),
        "bucket_size"
    ]
    if len(best_bucket) > 0:
        elo_bucket_size = best_bucket.iloc[0]
    else:
        # Fallback: use bucket with highest cache hit rate
        elo_bucket_size = elo_bucket_results.loc[
            elo_bucket_results["cache_hit_rate"].idxmax(), "bucket_size"
        ]
    print(f"    Selected Elo bucket size: {elo_bucket_size}")
    
    # Compute optimal PP policy using DP
    # Two scenarios are computed:
    # 1. Opponent PP still available (worst case) - more conservative strategy
    # 2. Opponent PP already used - more aggressive strategy (can use PP more freely)
    
    print("    Computing optimal PP policy (DP)...")
    print("      Scenario 1: Opponent PP still available (worst case)")
    score_diff_clip = (-10, 10)
    pp_policy_df = compute_pp_policy_heatmap(
        ep_model, differential_classes, class_to_diff, 
        score_range=(-5, 5), elo_diff=0.0, opp_pp_avail=1,
        elo_bucket_size=elo_bucket_size,
        score_diff_clip=score_diff_clip,
        early_quit_model=early_quit_model,
        extra_end_ep_model=None,  # Use regular EP model with EndsRemaining=0
        extra_end_differential_classes=None,
        extra_end_class_to_diff=None
    )
    plot_pp_policy_heatmap(
        pp_policy_df,
        save_path=os.path.join(results_dir, "pp_heatmap.png")
    )
    print("    Optimal PP policy heatmap saved: pp_heatmap.png")
    print("      (shows when to use PP assuming opponent still has PP available)")
    
    print("    Computing optimal PP policy (DP)...")
    print("      Scenario 2: Opponent PP already used")
    pp_policy_df_opp_used = compute_pp_policy_heatmap(
        ep_model, differential_classes, class_to_diff,
        score_range=(-5, 5), elo_diff=0.0, opp_pp_avail=0,
        elo_bucket_size=elo_bucket_size,
        score_diff_clip=score_diff_clip,
        early_quit_model=early_quit_model,
        extra_end_ep_model=None,  # Use regular EP model with EndsRemaining=0
        extra_end_differential_classes=None,
        extra_end_class_to_diff=None
    )
    plot_pp_policy_heatmap(
        pp_policy_df_opp_used,
        save_path=os.path.join(results_dir, "pp_heatmap_opp_used.png")
    )
    print("    Optimal PP policy heatmap saved: pp_heatmap_opp_used.png")
    print("      (shows when to use PP assuming opponent has already used their PP)")
    
    # Also compute simpler DeltaEP analysis for comparison
    print("    Computing DeltaEP analysis...")
    pp_decision_ends = end_level_df[
        (end_level_df["RefHasHammerStartOfEnd"] == 1) &
        (end_level_df["RefPPAvailableBeforeEnd"] == 1) &
        (end_level_df["EndID"].between(1, 8))
    ].copy()
    if len(pp_decision_ends) > 0:
        delta_ep_df = compute_pp_delta_ep(
            pp_decision_ends, ep_model, differential_classes, class_to_diff
        )
        delta_ep_agg = aggregate_decisions(delta_ep_df, value_col="DeltaEP")
        print(f"    Computed DeltaEP for {len(delta_ep_df):,} decision points")
    
    print()
    print("=" * 80)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

