"""
Results analysis and visualization for EP (Expected Points) model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

from ep_end import EP_FEATURE_COLS, prepare_end_level_features


def _format_ep_labels(differential_classes):
    """Format labels with tail bins for EP plots."""
    min_class = min(differential_classes)
    max_class = max(differential_classes)
    labels = []
    for diff in differential_classes:
        if diff == min_class and min_class <= -4:
            labels.append(f"{min_class}-")
        elif diff == max_class and max_class >= 4:
            labels.append(f"{max_class}+")
        else:
            labels.append(str(diff))
    return labels


def evaluate_ep_model(model, X_val, y_val, differential_classes, class_to_diff):
    """
    Evaluate EP distribution model performance.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained multiclass classifier
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation class labels (0-indexed)
    differential_classes : np.ndarray
        Array of differential values
    class_to_diff : dict
        Mapping from class index to differential value
    
    Returns
    -------
    dict
        Dictionary with accuracy, log_loss, and expected value error
    """
    # Predict probabilities
    val_probs = model.predict_proba(X_val)
    val_pred = model.predict(X_val)
    
    # Convert class labels back to differentials
    val_pred_diffs = pd.Series(val_pred).map(class_to_diff).values
    val_true_diffs = pd.Series(y_val).map(class_to_diff).values
    
    # Accuracy (exact match)
    accuracy = accuracy_score(y_val, val_pred)
    
    # Log loss
    logloss = log_loss(y_val, val_probs, labels=range(len(differential_classes)))
    
    # Expected value error (mean absolute error in expected differential)
    expected_pred = np.sum(val_probs * np.array([class_to_diff[i] for i in range(len(differential_classes))]), axis=1)
    expected_true = val_true_diffs
    mae_expected = np.mean(np.abs(expected_pred - expected_true))
    
    return {
        "accuracy": accuracy,
        "logloss": logloss,
        "mae_expected": mae_expected
    }


def plot_ep_confusion_matrix(model, X_val, y_val, differential_classes, class_to_diff, save_path=None):
    """
    Plot confusion matrix for EP distribution model.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained multiclass classifier
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation class labels
    differential_classes : np.ndarray
        Array of differential values
    class_to_diff : dict
        Mapping from class index to differential value
    save_path : str, optional
        Path to save the plot
    """
    val_pred = model.predict(X_val)
    
    # Convert to differentials for display
    val_pred_diffs = pd.Series(val_pred).map(class_to_diff).values
    val_true_diffs = pd.Series(y_val).map(class_to_diff).values
    
    cm = confusion_matrix(val_true_diffs, val_pred_diffs, labels=differential_classes)
    display_labels = _format_ep_labels(differential_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=display_labels, 
                yticklabels=display_labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted End Differential')
    plt.ylabel('Actual End Differential')
    plt.title('EP Model Confusion Matrix\n(End Differential Distribution)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ep_prediction_distribution(model, X_val, y_val, differential_classes, class_to_diff, save_path=None):
    """
    Plot predicted vs actual distribution of end differentials.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained multiclass classifier
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation class labels
    differential_classes : np.ndarray
        Array of differential values
    class_to_diff : dict
        Mapping from class index to differential value
    save_path : str, optional
        Path to save the plot
    """
    val_probs = model.predict_proba(X_val)
    val_true_diffs = pd.Series(y_val).map(class_to_diff).values
    
    # Compute predicted distribution (average probabilities)
    pred_dist = val_probs.mean(axis=0)
    
    # Compute actual distribution
    actual_dist = np.zeros(len(differential_classes))
    for i, diff in enumerate(differential_classes):
        actual_dist[i] = np.mean(val_true_diffs == diff)
    
    # Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(differential_classes))
    width = 0.35
    display_labels = _format_ep_labels(differential_classes)
    
    # Use a colorblind-friendly, high-contrast pair (avoid blue-green combo)
    plt.bar(x - width/2, actual_dist, width, label='Actual', color='#4C78A8', alpha=0.9)
    plt.bar(x + width/2, pred_dist, width, label='Predicted', color='#F58518', alpha=0.9)
    
    plt.xlabel('End Differential')
    plt.ylabel('Probability')
    plt.title('EP Model: Predicted vs Actual Distribution')
    plt.xticks(x, display_labels)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ep_feature_importance(model, save_path=None):
    """
    Plot feature importance for EP model.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained EP model
    save_path : str, optional
        Path to save the plot
    """
    importance = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': EP_FEATURE_COLS,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, y='feature', x='importance', color='steelblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('EP Model Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return feature_importance_df
