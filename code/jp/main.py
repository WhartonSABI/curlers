"""
Main script to run the complete curling win probability pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_prep import prepare_modeling_data
from model import train_model, FEATURE_COLS, TARGET_COL
from results import (
    evaluate_model,
    plot_calibration_curve,
    plot_feature_importance,
    get_decision_points2,
    compute_delta_wp,
    normalize_delta_wp,
    aggregate_decisions,
    plot_decision_heatmap
)


def main():
    """Main pipeline execution."""
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    results_dir = os.path.join(project_root, "results")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("Curling Win Probability Model Pipeline")
    print("=" * 80)
    print()
    
    # Step 1: Data Preparation
    print("Step 1: Loading and preparing data...")
    nnshots, games = prepare_modeling_data(data_dir=data_dir)
    print(f"  - Loaded {len(nnshots):,} shots")
    print(f"  - Loaded {len(games):,} games")
    
    # Export processed data
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    nnshots.to_csv(os.path.join(processed_dir, "processed_shots.csv"), index=False)
    games.to_csv(os.path.join(processed_dir, "processed_games.csv"), index=False)
    print(f"  - Processed data exported to {processed_dir}")
    print()
    
    # Step 2: Model Training
    print("Step 2: Training model...")
    model, train_df, val_df, X_train, X_val, y_train, y_val, le_ref_team, le_opp_team = train_model(nnshots)
    print("  - Model trained successfully")
    print()
    
    # Step 3: Model Evaluation
    print("Step 3: Evaluating model...")
    metrics = evaluate_model(model, X_val, y_val)
    print(f"  - Brier score: {metrics['brier_score']:.4f}")
    print(f"  - Log loss:    {metrics['logloss']:.4f}")
    print(f"  - Accuracy:    {metrics['accuracy']:.3f}")
    print()
    
    # Step 4: Generate Plots
    print("Step 4: Generating plots...")
    
    # Calibration curve
    print("  - Calibration curve...")
    plot_calibration_curve(
        model, X_val, y_val,
        save_path=os.path.join(results_dir, "calibration_curve.png")
    )
    
    # Feature importance
    print("  - Feature importance...")
    feature_importance_df = plot_feature_importance(
        model,
        save_path=os.path.join(results_dir, "feature_importance.png")
    )
    print(f"    Top 5 features:")
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    # Decision analysis
    print("  - Decision analysis...")
    decision_rows_df = get_decision_points2(nnshots)
    print(f"    Found {len(decision_rows_df)} decision points")
    
    decision_win_probs = compute_delta_wp(decision_rows_df, model, le_ref_team, le_opp_team)
    normalized_deltas = normalize_delta_wp(decision_win_probs)
    agg_decisions = aggregate_decisions(normalized_deltas)
    
    plot_decision_heatmap(
        agg_decisions,
        save_path=os.path.join(results_dir, "decision_heatmap.png")
    )
    
    print()
    print("=" * 80)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

