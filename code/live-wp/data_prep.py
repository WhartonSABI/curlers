"""
Data preparation for win probability model (shot-level features).
"""

import pandas as pd
import numpy as np
import os
import sys

# Import shared data prep functions from jp
jp_data_prep_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jp")
if jp_data_prep_path not in sys.path:
    sys.path.insert(0, jp_data_prep_path)
import data_prep as jp_data_prep
# Import constants (not needed but kept for reference)
# BUTTON_X, BUTTON_Y, HOUSE_RADIUS, etc. are in jp_data_prep

from elo import compute_elo_ratings


def add_elo_diff_feature(nnshots, games):
    """
    Add RefEloDiff feature to shots data.
    
    RefEloDiff = RefTeamElo - OppTeamElo (from RefTeam's perspective)
    Positive = RefTeam is stronger
    Negative = OppTeam is stronger
    
    Parameters
    ----------
    nnshots : pd.DataFrame
        Shots dataframe with RefTeamID and OppTeamID columns
    games : pd.DataFrame
        Games dataframe with game results
    
    Returns
    -------
    pd.DataFrame
        Shots dataframe with RefEloDiff column added
    """
    nnshots = nnshots.copy()
    
    # Compute Elo ratings from all games
    elo_ratings = compute_elo_ratings(games)
    
    # Add Elo ratings to shots
    nnshots['RefTeamElo'] = nnshots['RefTeamID'].map(elo_ratings).fillna(1500.0)
    nnshots['OppTeamElo'] = nnshots['OppTeamID'].map(elo_ratings).fillna(1500.0)
    
    # Calculate Elo difference from RefTeam's perspective
    nnshots['RefEloDiff'] = nnshots['RefTeamElo'] - nnshots['OppTeamElo']
    
    # Drop intermediate columns
    nnshots = nnshots.drop(columns=['RefTeamElo', 'OppTeamElo'])
    
    return nnshots


def prepare_modeling_data(data_dir="data"):
    """
    Main function to prepare shot-level data for win probability modeling.
    
    Returns
    -------
    nnshots : pd.DataFrame
        Processed dataframe ready for WP modeling
    games : pd.DataFrame
        Games dataframe
    """
    stones, teams, games, ends, competitors, competition = jp_data_prep.load_data(data_dir)
    
    ends_prep = jp_data_prep.prepare_ends(ends)
    shots = jp_data_prep.prepare_shots(stones, ends_prep)
    
    nn_features = ["CompetitionID", "SessionID", "GameID", "EndID", "TeamID", "TeamPlayerID", "MatchID", 
                   "Task", "Result", "PowerPlay", "TeamShotsRemainingInEnd", "OpponentShotsRemainingInEnd", 
                   "ScoreDiffStartOfEnd", "OpponentResult", "EndDifferential", "EndScoringOutcome", "HasHammer",
                   "stone_1_x", "stone_1_y", "stone_2_x", "stone_2_y", "stone_3_x", "stone_3_y", 
                   "stone_4_x", "stone_4_y", "stone_5_x", "stone_5_y", "stone_6_x", "stone_6_y",
                   "stone_7_x", "stone_7_y", "stone_8_x", "stone_8_y", "stone_9_x", "stone_9_y", 
                   "stone_10_x", "stone_10_y", "stone_11_x", "stone_11_y", "stone_12_x", "stone_12_y"]
    
    nnshots = shots[nn_features]
    nnshots = nnshots.merge(
        games[["CompetitionID", "SessionID", "GameID", "TeamID1", "TeamID2"]], 
        on=["CompetitionID", "SessionID", "GameID"], 
        how="left"
    )
    
    nnshots = nnshots.dropna()
    nnshots = nnshots.reset_index(drop=True)
    
    nnshots = jp_data_prep.assign_stone_ownership(nnshots)
    nnshots = jp_data_prep.add_stone_distances(nnshots)
    nnshots = jp_data_prep.add_stones_in_house(nnshots)
    nnshots = jp_data_prep.add_closest_stone_features(nnshots)
    nnshots = jp_data_prep.add_crowding_features(nnshots)
    nnshots = jp_data_prep.add_power_play_features(nnshots)
    nnshots = jp_data_prep.add_reference_team_features(nnshots, games)
    nnshots = add_elo_diff_feature(nnshots, games)
    nnshots = jp_data_prep.add_game_state_features(nnshots)
    nnshots = jp_data_prep.add_final_state_rows(nnshots, ends_prep)
    
    return nnshots, games

