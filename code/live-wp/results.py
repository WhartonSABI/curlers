"""
Results analysis and visualization for win probability model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.calibration import calibration_curve

from wp_model import FEATURE_COLS, prepare_features


def plot_feature_importance(model, save_path=None):
    """
    Plot feature importance for XGBoost model (works with calibrated or uncalibrated models).
    
    Parameters
    ----------
    model : XGBClassifier or CalibratedClassifierCV
        Trained XGBoost model (calibrated or uncalibrated)
    save_path : str, optional
        Path to save the plot. If None, plot is shown instead.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with feature names and importance values
    """
    # Get feature importance (handle both calibrated and uncalibrated models)
    if hasattr(model, 'calibrated_classifiers_'):
        # Calibrated model - get importance from base estimator in first calibrated classifier
        base_model = model.calibrated_classifiers_[0].estimator
        importance = base_model.feature_importances_
    else:
        # Uncalibrated model
        importance = model.feature_importances_
    
    # Create dataframe
    feature_importance_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, y='feature', x='importance', color='steelblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Win Probability Model Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return feature_importance_df


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model performance metrics.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained model
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    
    Returns
    -------
    dict
        Dictionary with brier_score, logloss, and accuracy
    """
    val_probs = model.predict_proba(X_val)[:, 1]
    
    brier = brier_score_loss(y_val, val_probs)
    logloss = log_loss(y_val, val_probs)
    acc = accuracy_score(y_val, val_probs >= 0.5)
    
    return {
        "brier_score": brier,
        "logloss": logloss,
        "accuracy": acc
    }


def plot_calibration_curve(model, X_val, y_val, n_bins=10, save_path=None):
    """
    Plot calibration curve for model predictions.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained model
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    n_bins : int, default=10
        Number of bins for calibration curve
    save_path : str, optional
        Path to save the plot. If None, plot is shown instead.
    """
    val_probs = model.predict_proba(X_val)[:, 1]
    
    prob_true, prob_pred = calibration_curve(
        y_val,
        val_probs,
        n_bins=n_bins,
        strategy="uniform"
    )
    
    plt.figure(figsize=(7, 6))
    
    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=2,
        label="XGBoost"
    )
    
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    
    plt.xlabel("Predicted win probability")
    plt.ylabel("Observed win frequency")
    plt.title("Calibration Curve (Validation Set)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_win_probability(matchID, df, model, le_ref_team, le_opp_team, save_path=None):
    """
    Plot win probability over time for a specific match.
    
    Parameters
    ----------
    matchID : int
        Match ID to plot
    df : pd.DataFrame
        Dataframe with match data
    model : XGBClassifier
        Trained model
    le_ref_team : LabelEncoder
        Fitted encoder for RefTeamID
    le_opp_team : LabelEncoder
        Fitted encoder for OppTeamID
    save_path : str, optional
        Path to save the plot. If None, plot is shown instead.
    """
    gamedata = df[df["MatchID"] == matchID].copy()
    
    if len(gamedata) == 0:
        print(f"No data found for MatchID {matchID}")
        return
    
    gamedata = gamedata.reset_index(drop=True)
    gamedata["ShotIndex"] = np.arange(len(gamedata))
    
    X, _, _ = prepare_features(gamedata, le_ref_team=le_ref_team, le_opp_team=le_opp_team, fit_encoders=False)
    gamedata["RefWinProb"] = model.predict_proba(X)[:, 1]
    gamedata["OppWinProb"] = 1.0 - gamedata["RefWinProb"]
    
    # Automatically set to 100%/0% when game is over based on score
    # Game is over when both shots remaining = 0 and no ends remaining (or it's the last shot)
    game_over_mask = (
        (gamedata["RefShotsRemaining"] == 0) & 
        (gamedata["OppShotsRemaining"] == 0) &
        (gamedata["EndsRemaining"] == 0)
    )
    
    # If no ends remaining but game_over_mask is empty, check if it's the last shot of the game
    if not game_over_mask.any() and len(gamedata) > 0:
        # Check if last row has both shots remaining = 0
        last_row = gamedata.iloc[-1]
        if (last_row["RefShotsRemaining"] == 0) and (last_row["OppShotsRemaining"] == 0):
            game_over_mask = gamedata.index == gamedata.index[-1]
    
    # For shots where game is over, set probability based on score differential
    if game_over_mask.any():
        for idx in gamedata[game_over_mask].index:
            score_diff = gamedata.loc[idx, "RefScoreDiff"]
            if score_diff > 0:
                gamedata.loc[idx, "RefWinProb"] = 1.0
                gamedata.loc[idx, "OppWinProb"] = 0.0
            elif score_diff < 0:
                gamedata.loc[idx, "RefWinProb"] = 0.0
                gamedata.loc[idx, "OppWinProb"] = 1.0
            else:
                # Score differential = 0 (tie) - keep model prediction or use actual outcome
                if "RefTeamWon" in gamedata.columns:
                    final_outcome = gamedata.loc[idx, "RefTeamWon"]
                    gamedata.loc[idx, "RefWinProb"] = float(final_outcome)
                    gamedata.loc[idx, "OppWinProb"] = 1.0 - float(final_outcome)
    
    ref_team = gamedata["RefTeamID"].iloc[0]
    opp_team = gamedata["OppTeamID"].iloc[0]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(
        gamedata["ShotIndex"],
        gamedata["RefWinProb"],
        label=f"Ref Team {ref_team}",
        linewidth=2
    )
    
    plt.plot(
        gamedata["ShotIndex"],
        gamedata["OppWinProb"],
        label=f"Opp Team {opp_team}",
        linewidth=2
    )
    
    plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    plt.ylim(0, 1)
    
    plt.xlabel("Shot (Game Timeline)")
    plt.ylabel("Win Probability")
    plt.title(f"Shot-Level Win Probability (MatchID = {matchID})")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

