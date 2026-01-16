"""
Expected points model for curling end outcomes.

This module trains an end-level distributional EP model to predict P(Δ=k)
for end differentials. The model operates on start-of-end states, which
is the correct decision-time for power play decisions.

Key changes from shot-level model:
- One row per match-end at start of end (not shot-level)
- Features are decision-time only (EndsRemaining, score diff, hammer, PP availability)
- Predicts full distribution P(Δ=k) not just mean
- Used for optimal PP timing via dynamic programming
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

# EP model features: Decision-time only, from ref team's perspective
EP_FEATURE_COLS = [
    "EndsRemaining",  # Ends remaining after current end: max(8 - EndID, 0)
    "RefHasHammerStartOfEnd",  # 1 if ref has hammer, 0 otherwise
    "RefScoreDiffStartOfEnd",  # Score diff from ref perspective at start
    "PPUsedThisEnd",  # 1 if ref uses PP, -1 if opp uses PP, 0 if neither
    "RefPPAvailableBeforeEnd",  # 1 if ref PP available, 0 otherwise
    "OppPPAvailableBeforeEnd",  # 1 if opp PP available, 0 otherwise
    "RefEloDiff",  # Optional: Elo difference (ref - opp)
]


def prepare_end_level_features(end_df):
    """
    Prepare feature matrix for end-level EP model.
    
    Parameters
    ----------
    end_df : pd.DataFrame
        End-level dataframe from build_start_of_end_df()
        Must have EndID column to compute EndsRemaining
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix with EP_FEATURE_COLS
    """
    # Compute EndsRemaining if not already present
    if "EndsRemaining" not in end_df.columns:
        if "EndID" not in end_df.columns:
            raise ValueError("Must have either EndsRemaining or EndID column")
        end_df = end_df.copy()
        end_df["EndsRemaining"] = np.maximum(8 - end_df["EndID"], 0)
    
    # Ensure all required columns exist
    missing_cols = [col for col in EP_FEATURE_COLS if col not in end_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    X = end_df[EP_FEATURE_COLS].copy()
    return X


def train_end_differential_distribution_model(
    end_df,
    test_size=0.2,
    random_state=8,
    is_extra_end=False,
    use_class_weights=False,
    class_weight_power=1.0
):
    """
    Train distributional model to predict P(Δ=k) for end differential.
    
    Predicts probability distribution over end differentials from ref team's perspective:
    - Positive: ref team scores (1, 2, 3, etc.)
    - Negative: opponent scores (-1, -2, -3, etc.)
    - Zero: blank end
    
    Parameters
    ----------
    end_df : pd.DataFrame
        End-level dataframe from build_start_of_end_df()
        Must have RefEndDifferential column
    test_size : float, default=0.2
        Fraction of data for validation
    random_state : int, default=8
        Random seed for train/test split
    is_extra_end : bool, default=False
        If True, train model specifically for extra ends (EndID > 8)
    use_class_weights : bool, default=False
        If True, use inverse-frequency class weights to emphasize rare outcomes
    class_weight_power : float, default=1.0
        Exponent for inverse-frequency weights (1.0 = full weighting, 0.0 = no weighting)
    
    Returns
    -------
    model : XGBClassifier
        Trained multiclass classifier for end differential distribution
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    differential_classes : np.ndarray
        Array of possible differential values (class labels)
    class_to_diff : dict
        Mapping from class index to differential value
    """
    # Filter for extra ends if specified
    if is_extra_end:
        end_df = end_df[end_df["EndID"] > 8].copy()
        if len(end_df) == 0:
            raise ValueError("No extra end data available for training")
    else:
        # Regular ends: EndID <= 8
        end_df = end_df[end_df["EndID"] <= 8].copy()
    
    # Create MatchID for grouped split
    if "MatchID" not in end_df.columns:
        end_df = end_df.copy()
        end_df["MatchID"] = (
            end_df["CompetitionID"].astype(str) + "_" +
            end_df["SessionID"].astype(str) + "_" +
            end_df["GameID"].astype(str)
        )
    
    # Use grouped split by MatchID to avoid data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(end_df, groups=end_df["MatchID"]))
    train_df = end_df.iloc[train_idx].copy()
    val_df = end_df.iloc[val_idx].copy()
    
    # Get range of differentials in training data
    min_diff = int(train_df["RefEndDifferential"].min())
    max_diff = int(train_df["RefEndDifferential"].max())
    
    # Handle class imbalance: use coarser bins for rare outcomes
    # Map extreme outcomes to bins: ≤-4, -3, -2, -1, 0, +1, +2, +3, ≥+4
    # This stabilizes tail predictions which are critical for PP value
    if min_diff < -4 or max_diff > 4:
        # Use binned classes for better tail handling
        differential_classes = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        # Clip and map extreme values to bins
        train_df["EndDiffBinned"] = train_df["RefEndDifferential"].clip(-4, 4).astype(int)
        val_df["EndDiffBinned"] = val_df["RefEndDifferential"].clip(-4, 4).astype(int)
        train_df["EndDiffClass"] = train_df["EndDiffBinned"]
        val_df["EndDiffClass"] = val_df["EndDiffBinned"]
    else:
        # Use exact differentials if range is small
        differential_classes = np.arange(max(-6, min_diff), min(7, max_diff + 1))
        train_df["EndDiffClass"] = train_df["RefEndDifferential"].clip(
            lower=differential_classes.min(), 
            upper=differential_classes.max()
        ).astype(int)
        val_df["EndDiffClass"] = val_df["RefEndDifferential"].clip(
            lower=differential_classes.min(),
            upper=differential_classes.max()
        ).astype(int)
    
    # Map differentials to 0-indexed class labels (XGBoost requirement)
    diff_to_class = {diff: idx for idx, diff in enumerate(differential_classes)}
    class_to_diff = {idx: diff for diff, idx in diff_to_class.items()}
    
    train_df["ClassLabel"] = train_df["EndDiffClass"].map(diff_to_class)
    val_df["ClassLabel"] = val_df["EndDiffClass"].map(diff_to_class)
    
    X_train = prepare_end_level_features(train_df)
    X_val = prepare_end_level_features(val_df)
    
    y_train = train_df["ClassLabel"]
    y_val = val_df["ClassLabel"]
    
    sample_weights = None
    if use_class_weights and class_weight_power > 0:
        # Inverse frequency weighting: rare classes get higher weight
        class_counts = train_df["ClassLabel"].value_counts().sort_index()
        total_samples = len(train_df)
        class_weights = {}
        for class_idx in range(len(differential_classes)):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            base_weight = total_samples / (len(differential_classes) * count)
            class_weights[class_idx] = base_weight ** class_weight_power
        
        # Convert to sample_weight array
        sample_weights = train_df["ClassLabel"].map(class_weights).values
    
    mono_constraints = [0] * len(EP_FEATURE_COLS)
    if "RefEloDiff" in EP_FEATURE_COLS:
        mono_constraints[EP_FEATURE_COLS.index("RefEloDiff")] = 1
    monotone_constraints = "(" + ",".join(str(v) for v in mono_constraints) + ")"

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=19,
        objective="multi:softprob",
        num_class=len(differential_classes),
        monotone_constraints=monotone_constraints,
        scale_pos_weight=1.0  # Will use sample_weights instead
    )
    
    fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights

    model.fit(X_train, y_train, **fit_kwargs)
    
    return model, train_df, val_df, differential_classes, class_to_diff


def predict_end_differential_distribution(row, model, differential_classes, class_to_diff, is_extra_end=False):
    """
    Predict probability distribution over end differentials from a game state.
    
    Parameters
    ----------
    row : pd.Series
        Game state row (must have EP_FEATURE_COLS)
    model : XGBClassifier
        Trained multiclass classifier
    differential_classes : np.ndarray
        Array of possible differential values
    class_to_diff : dict
        Mapping from class index to differential value
    is_extra_end : bool, default=False
        If True, this is an extra end prediction
    
    Returns
    -------
    dict
        Dictionary mapping differential value to probability: {diff: prob, ...}
    """
    row_df = pd.DataFrame([row])
    X = prepare_end_level_features(row_df)
    
    # Get probability distribution (over class indices)
    probs = model.predict_proba(X)[0]
    
    # Map class indices back to differential values
    prob_dist = {int(class_to_diff[i]): float(prob) for i, prob in enumerate(probs)}
    
    return prob_dist


def train_early_quit_model(end_df, test_size=0.2, random_state=8):
    """
    Train model to predict probability of early game termination (concession).
    
    Parameters
    ----------
    end_df : pd.DataFrame
        End-level dataframe from build_start_of_end_df()
        Must have EarlyQuit column
    test_size : float, default=0.2
        Fraction of data for validation
    random_state : int, default=8
        Random seed for train/test split
    
    Returns
    -------
    model : XGBClassifier
        Trained binary classifier for early quit probability
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    # Filter to ends where early quit is possible (not already quit)
    # Only consider ends 1-7 (can't quit after end 8)
    end_df = end_df[end_df["EndID"] <= 7].copy()
    
    # Create MatchID for grouped split
    if "MatchID" not in end_df.columns:
        end_df = end_df.copy()
        end_df["MatchID"] = (
            end_df["CompetitionID"].astype(str) + "_" +
            end_df["SessionID"].astype(str) + "_" +
            end_df["GameID"].astype(str)
        )
    
    # Use grouped split by MatchID to avoid data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(end_df, groups=end_df["MatchID"]))
    train_df = end_df.iloc[train_idx].copy()
    val_df = end_df.iloc[val_idx].copy()
    
    # Features for early quit prediction
    early_quit_features = [
        "EndID",
        "RefHasHammerStartOfEnd",
        "RefScoreDiffStartOfEnd",
        "RefPPAvailableBeforeEnd",
        "OppPPAvailableBeforeEnd",
        "RefEloDiff"
    ]
    
    X_train = train_df[early_quit_features].copy()
    X_val = val_df[early_quit_features].copy()
    y_train = train_df["EarlyQuit"]
    y_val = val_df["EarlyQuit"]
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=19,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model, train_df, val_df


def predict_early_quit_probability(row, model):
    """
    Predict probability of early game termination after this end.
    
    Parameters
    ----------
    row : pd.Series
        Game state row (must have early quit features)
    model : XGBClassifier
        Trained early quit model
    
    Returns
    -------
    float
        Probability of early quit (0-1)
    """
    early_quit_features = [
        "EndID",
        "RefHasHammerStartOfEnd",
        "RefScoreDiffStartOfEnd",
        "RefPPAvailableBeforeEnd",
        "OppPPAvailableBeforeEnd",
        "RefEloDiff"
    ]
    
    row_df = pd.DataFrame([row])
    X = row_df[early_quit_features].copy()
    
    prob = model.predict_proba(X)[0][1]  # Probability of class 1 (early quit)
    return float(prob)
