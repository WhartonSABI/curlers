"""
Decision analysis for curling win probability model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import prepare_features


def get_decision_points(df):
    """
    Select decision points where the hammer team has exactly
    one shot remaining and the non-hammer team has zero.
    Works for both ref-hammer and opp-hammer situations.
    """
    mask_ref_hammer = (
        (df["RefHasHammer"] == 1) &
        (df["RefShotsRemaining"] == 1) &
        (df["OppShotsRemaining"] == 0)
    )
    
    mask_opp_hammer = (
        (df["RefHasHammer"] == 0) &
        (df["RefShotsRemaining"] == 0) &
        (df["OppShotsRemaining"] == 1)
    )
    
    return df[
        (df["EndID"].between(3, 7)) &
        (mask_ref_hammer | mask_opp_hammer)
    ].copy()


def create_counterfactuals(row):
    """Create score and punt counterfactual scenarios."""
    base = row.copy()
    score = base.copy()
    
    if row["RefHasHammer"] == 1:
        score["RefScoreDiff"] += 1
    else:
        score["RefScoreDiff"] -= 1
    score["EndID"] += 1
    score["EndsRemaining"] -= 1
    score["RefShotsRemaining"] = 5
    score["OppShotsRemaining"] = 5
    score["PowerPlay"] = 0
    
    for col in score.index:
        if "stone_" in col:
            score[col] = 0
    
    if score["RefHasHammer"] == 1:
        score["RefClosestDist"] = 1116
        score["OppClosestDist"] = 150
        score["RefStonesInHouse"] = 0
        score["OppStonesInHouse"] = 1
        score["RefStonesCloserDiff"] = -1
        score["StonesCrowdedNearButton"] = 1
        score["StonesMediumCrowdedNearButton"] = 1
        if base["PowerPlay"] == 1:
            score["RefHasUsedPowerPlay"] = 1
    else:
        score["RefClosestDist"] = 150
        score["OppClosestDist"] = 1116
        score["RefStonesInHouse"] = 1
        score["OppStonesInHouse"] = 0
        score["RefStonesCloserDiff"] = 1
        score["StonesCrowdedNearButton"] = 1
        score["StonesMediumCrowdedNearButton"] = 1
        if base["PowerPlay"] == 1:
            score["OppHasUsedPowerPlay"] = 1
    
    score["RefHasHammer"] = 1 - base["RefHasHammer"]
    
    punt = base.copy()
    punt["RefHasHammer"] = base["RefHasHammer"]
    punt["EndID"] += 1
    punt["EndsRemaining"] -= 1
    punt["RefShotsRemaining"] = 5
    punt["OppShotsRemaining"] = 5
    punt["PowerPlay"] = 0
    
    for col in punt.index:
        if "stone_" in col:
            punt[col] = 0
    
    if punt["RefHasHammer"] == 1:
        punt["RefClosestDist"] = 150
        punt["OppClosestDist"] = 1116
        punt["RefStonesInHouse"] = 1
        punt["OppStonesInHouse"] = 0
        punt["RefStonesCloserDiff"] = 1
        punt["StonesCrowdedNearButton"] = 1
        punt["StonesMediumCrowdedNearButton"] = 1
        if base["PowerPlay"] == 1:
            punt["RefHasUsedPowerPlay"] = 1
    else:
        punt["RefClosestDist"] = 1116
        punt["OppClosestDist"] = 150
        punt["RefStonesInHouse"] = 0
        punt["OppStonesInHouse"] = 1
        punt["RefStonesCloserDiff"] = -1
        punt["StonesCrowdedNearButton"] = 1
        punt["StonesMediumCrowdedNearButton"] = 1
        if base["PowerPlay"] == 1:
            punt["OppHasUsedPowerPlay"] = 1
    
    return score, punt


def compute_delta_wp(df, model, le_ref_team, le_opp_team):
    """
    Compute DeltaWP = WinProb(punt) - WinProb(score)
    
    Positive DeltaWP: Punting is better than scoring (should punt/blank the end)
    Negative DeltaWP: Scoring is better than punting (should score)
    """
    rows = []
    for _, row in df.iterrows():
        score_row, punt_row = create_counterfactuals(row)
        
        score_df = pd.DataFrame([score_row])
        punt_df = pd.DataFrame([punt_row])
        
        X_score, _, _ = prepare_features(score_df, le_ref_team=le_ref_team, le_opp_team=le_opp_team, fit_encoders=False)
        X_punt, _, _ = prepare_features(punt_df, le_ref_team=le_ref_team, le_opp_team=le_opp_team, fit_encoders=False)
        
        wp_score = model.predict_proba(X_score)[0, 1]
        wp_punt = model.predict_proba(X_punt)[0, 1]
        
        delta_wp = wp_punt - wp_score
        rows.append({
            "EndID": row["EndID"], 
            "RefScoreDiff": row["RefScoreDiff"], 
            "RefHasHammer": row["RefHasHammer"], 
            "DeltaWP": delta_wp
        })
    
    return pd.DataFrame(rows)


def normalize_delta_wp(df):
    """
    Normalize DeltaWP to hammer team's perspective.
    
    DeltaWP = WinProb(punt) - WinProb(score)
    HammerDeltaWP: From hammer team's perspective
      - If hammer team: HammerDeltaWP = DeltaWP
      - If non-hammer team: HammerDeltaWP = -DeltaWP (flip perspective)
    
    Positive HammerDeltaWP: Hammer team should punt (blank)
    Negative HammerDeltaWP: Hammer team should score
    """
    df = df.copy()
    df["HammerDeltaWP"] = df.apply(
        lambda r: r["DeltaWP"] if r["RefHasHammer"] == 1 else -r["DeltaWP"],
        axis=1)
    return df


def aggregate_decisions(df):
    """Aggregate decisions by EndID and RefScoreDiff."""
    return df.groupby(["EndID", "RefScoreDiff"]).agg(
        mean_delta_wp=("HammerDeltaWP", "mean"), 
        count=("HammerDeltaWP", "size")
    ).reset_index()


def plot_decision_heatmap(df, save_path=None):
    """Plot decision heatmap."""
    pivot = df.pivot(index="RefScoreDiff", columns="EndID", values="mean_delta_wp")
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="RdBu", center=0, annot=True, fmt=".3f")
    plt.title("ΔWP = WinProb(punt) - WinProb(score)\nPositive = Should Punt, Negative = Should Score")
    plt.xlabel("End")
    plt.ylabel("Score Differential (Ref Team)")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

