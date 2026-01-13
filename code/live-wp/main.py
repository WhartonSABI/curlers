"""
Main script to train and evaluate the win probability model.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to path (must be first to import local modules)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import local modules explicitly
import importlib.util

# Import data_prep
data_prep_spec = importlib.util.spec_from_file_location("data_prep", os.path.join(current_dir, "data_prep.py"))
data_prep = importlib.util.module_from_spec(data_prep_spec)
data_prep_spec.loader.exec_module(data_prep)
prepare_modeling_data = data_prep.prepare_modeling_data

# Import wp_model
wp_model_spec = importlib.util.spec_from_file_location("wp_model", os.path.join(current_dir, "wp_model.py"))
wp_model = importlib.util.module_from_spec(wp_model_spec)
wp_model_spec.loader.exec_module(wp_model)
train_model = wp_model.train_model
FEATURE_COLS = wp_model.FEATURE_COLS
TARGET_COL = wp_model.TARGET_COL

# Import results
results_spec = importlib.util.spec_from_file_location("results", os.path.join(current_dir, "results.py"))
results = importlib.util.module_from_spec(results_spec)
results_spec.loader.exec_module(results)
evaluate_model = results.evaluate_model
plot_calibration_curve = results.plot_calibration_curve
plot_feature_importance = results.plot_feature_importance
plot_win_probability = results.plot_win_probability


def main():
    """Main pipeline execution for win probability model."""
    
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "raw")
    results_dir = os.path.join(project_root, "results", "live-wp")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 80)
    print("Win Probability Model Pipeline")
    print("=" * 80)
    print()
    
    # Step 1: Data Preparation
    print("Step 1: Loading and preparing data...")
    nnshots, games = prepare_modeling_data(data_dir=data_dir)
    print(f"  - Loaded {len(nnshots):,} shots")
    print(f"  - Loaded {len(games):,} games")
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
        save_path=os.path.join(results_dir, "wp_calibration_curve.png")
    )
    
    # Feature importance
    print("  - Feature importance...")
    feature_importance_df = plot_feature_importance(
        model,
        save_path=os.path.join(results_dir, "wp_feature_importance.png")
    )
    print(f"    Top 5 features:")
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    print()
    print("=" * 80)
    print("Pipeline completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)
    print()
    print("To plot win probability for a specific match, use:")
    print("  from results import plot_win_probability")
    print("  plot_win_probability(matchID, nnshots, model, le_ref_team, le_opp_team)")


if __name__ == "__main__":
    main()

