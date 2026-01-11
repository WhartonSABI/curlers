"""
Model training for curling win probability prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


FEATURE_COLS = [
    "EndsRemaining", "EarlyGameEnd", "RefTeamID", "OppTeamID", 
    "RefHasUsedPowerPlay", "OppHasUsedPowerPlay", 
    "RefShotsRemaining", "OppShotsRemaining", 
    "RefHasHammer", "RefStonesInHouse", "OppStonesInHouse", 
    "RefClosestDist", "OppClosestDist", "RefStonesCloserDiff", 
    "StonesCrowdedNearButton", "StonesMediumCrowdedNearButton", "RefScoreDiff", "RefEloDiff"
]
TARGET_COL = "RefTeamWon"


def prepare_features(df, le_ref_team=None, le_opp_team=None, fit_encoders=True):
    """
    Prepare feature matrices with label encoding for categorical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features
    le_ref_team : LabelEncoder, optional
        Pre-fitted encoder for RefTeamID
    le_opp_team : LabelEncoder, optional
        Pre-fitted encoder for OppTeamID
    fit_encoders : bool, default=True
        Whether to fit new encoders or use provided ones
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    le_ref_team : LabelEncoder
        Fitted encoder for RefTeamID
    le_opp_team : LabelEncoder
        Fitted encoder for OppTeamID
    """
    X = df[FEATURE_COLS].copy()
    
    if fit_encoders or le_ref_team is None:
        le_ref_team = LabelEncoder()
        X["RefTeamID"] = le_ref_team.fit_transform(X["RefTeamID"].astype(str))
    else:
        X["RefTeamID"] = le_ref_team.transform(X["RefTeamID"].astype(str))
    
    if fit_encoders or le_opp_team is None:
        le_opp_team = LabelEncoder()
        X["OppTeamID"] = le_opp_team.fit_transform(X["OppTeamID"].astype(str))
    else:
        X["OppTeamID"] = le_opp_team.transform(X["OppTeamID"].astype(str))
    
    return X, le_ref_team, le_opp_team


def train_model(
    nnshots,
    test_size=0.2,
    random_state=8,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    model_random_state=19
):
    """
    Train XGBoost model for win probability prediction.
    
    Parameters
    ----------
    nnshots : pd.DataFrame
        Processed dataframe with features
    test_size : float, default=0.2
        Fraction of data for validation
    random_state : int, default=8
        Random seed for train/test split
    n_estimators : int, default=300
        Number of boosting rounds
    max_depth : int, default=4
        Maximum tree depth
    learning_rate : float, default=0.05
        Learning rate
    subsample : float, default=0.8
        Subsample ratio
    colsample_bytree : float, default=0.8
        Column subsample ratio
    eval_metric : str, default="logloss"
        Evaluation metric
    model_random_state : int, default=19
        Random seed for model
    
    Returns
    -------
    model : XGBClassifier
        Trained model
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame
        Validation features
    y_train : pd.Series
        Training target
    y_val : pd.Series
        Validation target
    le_ref_team : LabelEncoder
        Fitted encoder for RefTeamID
    le_opp_team : LabelEncoder
        Fitted encoder for OppTeamID
    """
    train_df, val_df = train_test_split(nnshots, test_size=test_size, random_state=random_state)
    
    X_train, le_ref_team, le_opp_team = prepare_features(train_df, fit_encoders=True)
    X_val, _, _ = prepare_features(val_df, le_ref_team=le_ref_team, le_opp_team=le_opp_team, fit_encoders=False)
    
    y_train = train_df[TARGET_COL]
    y_val = val_df[TARGET_COL]
    
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric=eval_metric,
        random_state=model_random_state
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Calibrate the model for better probability estimates
    # Use cross-validation to avoid overfitting the calibration mapping
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model, train_df, val_df, X_train, X_val, y_train, y_val, le_ref_team, le_opp_team

