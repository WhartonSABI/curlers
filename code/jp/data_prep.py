"""
Data preparation and feature engineering for curling shot-level data.
"""

import pandas as pd
import numpy as np
import os
from elo import compute_elo_ratings


BUTTON_X = 750
BUTTON_Y = 800
HOUSE_RADIUS = 600
CROWDED_RADIUS = 200
MEDIUM_CROWDED_RADIUS = 400


def load_data(data_dir="data"):
    """Load all CSV files."""
    stones = pd.read_csv(f"{data_dir}/Stones.csv", low_memory=False)
    teams = pd.read_csv(f"{data_dir}/Teams.csv", low_memory=False)
    games = pd.read_csv(f"{data_dir}/Games.csv", low_memory=False)
    ends = pd.read_csv(f"{data_dir}/Ends.csv", low_memory=False)
    competitors = pd.read_csv(f"{data_dir}/Competitors.csv", low_memory=False)
    competition = pd.read_csv(f"{data_dir}/Competition.csv", low_memory=False)
    return stones, teams, games, ends, competitors, competition


def prepare_ends(ends):
    """Process ends dataframe."""
    ends = ends.copy()
    ends["PowerPlay"] = ends["PowerPlay"].replace("", np.nan)
    ends["PowerPlay"] = pd.to_numeric(ends["PowerPlay"], errors="coerce")
    
    ends["TotalGameScoreAfterEnd"] = ends.groupby(["CompetitionID", "SessionID", "GameID", 
                                       "TeamID"])["Result"].cumsum()
    ends["TotalGameScoreStartOfEnd"] = ends["TotalGameScoreAfterEnd"] - ends["Result"]
    ends["OpponentScoreStartOfEnd"] = ends.groupby(["CompetitionID", "SessionID", "GameID", 
                                                "EndID"])["TotalGameScoreStartOfEnd"].transform("sum") - ends["TotalGameScoreStartOfEnd"]
    ends["ScoreDiffStartOfEnd"] = ends["TotalGameScoreStartOfEnd"] - ends["OpponentScoreStartOfEnd"]
    ends["OpponentResult"] = ends.groupby(["CompetitionID", "SessionID", "GameID", 
                                                "EndID"])["Result"].transform("sum") - ends["Result"]
    ends["EndDifferential"] = ends["Result"] - ends["OpponentResult"]
    ends["EndScoringOutcome"] = ends.apply(lambda x: 0 if x["Result"] > 0 else 1 if x["OpponentResult"] > 0 else 2, axis=1)
    
    return ends


def team_shot(df):
    """Calculate team shot number in end."""
    df = df.copy()
    df.loc[df["HasHammer"] == 0, "TeamShotNumberInEnd"] = (
        df.loc[df["HasHammer"] == 0, "TotalShotNumberInEnd"] // 2 + 1
    )
    df.loc[df["HasHammer"] == 1, "TeamShotNumberInEnd"] = (
        df.loc[df["HasHammer"] == 1, "TotalShotNumberInEnd"] // 2
    )
    df["TeamShotsRemainingInEnd"] = 5 - df["TeamShotNumberInEnd"]
    return df


def opponent_shots_remaining(df):
    """Calculate opponent shot number in end."""
    df = df.copy()
    df.loc[df["HasHammer"] == 1, "OpponentShotNumberInEnd"] = (
        df.loc[df["HasHammer"] == 1, "TotalShotNumberInEnd"] // 2
    )
    df.loc[df["HasHammer"] == 0, "OpponentShotNumberInEnd"] = (
        df.loc[df["HasHammer"] == 0, "TotalShotNumberInEnd"] // 2
    )
    df["OpponentShotsRemainingInEnd"] = 5 - df["OpponentShotNumberInEnd"]
    return df


def prepare_shots(stones, ends):
    """Process stones and merge with ends data."""
    # Merge with ends data (ends should already be prepared with all columns)
    shots = stones.merge(ends, on=["CompetitionID", "SessionID", "GameID", "TeamID", "EndID"], how="left")
    
    shots["TotalShotNumberInEnd"] = shots.groupby(["CompetitionID", "SessionID", "GameID", "EndID"]).cumcount()+1
    shots["TotalShotsRemainingInEnd"] = 10 - shots["TotalShotNumberInEnd"]
    shots["PowerPlay"] = shots["PowerPlay"].fillna(0)
    
    shots["HasHammer"] = shots["TotalShotNumberInEnd"].apply(lambda x: 1 if x in [2,4,6,8,10] else 0)
    
    shots["TeamPlayerID"] = (shots["TeamID"].astype(int).astype(str) 
                             + shots["PlayerID"].astype(int).astype(str)).astype(int)
    shots["MatchID"] = (shots["CompetitionID"].astype(int).astype(str) 
                             + shots["SessionID"].astype(int).astype(str) + shots["GameID"].astype(int).astype(str)).astype(int)
    
    shots = shots[(shots["Task"]>=0) & (shots["Task"]<=13)]
    shots = shots[shots["Result"] != 9]
    
    shots = team_shot(shots)
    shots = opponent_shots_remaining(shots)
    
    return shots


def is_invalid_stone(df, i):
    """Check if stone is invalid (not thrown or out of play)."""
    x = df[f"stone_{i}_x"]
    y = df[f"stone_{i}_y"]
    return ((x==0) & (y==0)) | ((x==4095) & (y==4095))


def assign_stone_ownership(nnshots):
    """Assign stone ownership based on TeamID1/TeamID2."""
    for i in range(1, 13):
        col_name = f"stone_{i}_is_teams"
        if i <= 6:
            nnshots[col_name] = (nnshots["TeamID"] == nnshots["TeamID1"]).astype(int)
        else:
            nnshots[col_name] = (nnshots["TeamID"] == nnshots["TeamID2"]).astype(int)
        invalid_mask = is_invalid_stone(nnshots, i)
        nnshots.loc[invalid_mask, col_name] = 0
    return nnshots


def add_stone_distances(nnshots):
    """Calculate distances from each stone to button."""
    for i in range(1, 13):
        x_col = f"stone_{i}_x"
        y_col = f"stone_{i}_y"
        d_col = f"stone_{i}_dist_to_button"
        nnshots[d_col] = np.sqrt(
            (nnshots[x_col] - BUTTON_X)**2 + 
            (nnshots[y_col] - BUTTON_Y)**2)
        invalid_mask = is_invalid_stone(nnshots, i)
        nnshots.loc[invalid_mask, d_col] = 4000
    return nnshots


def add_stones_in_house(df):
    """Mark stones in house and calculate aggregates."""
    df = df.copy()
    
    for i in range(1, 13):
        dx = df[f"stone_{i}_x"] - BUTTON_X
        dy = df[f"stone_{i}_y"] - BUTTON_Y
        df[f"stone_{i}_in_house"] = (
            np.sqrt(dx**2 + dy**2) <= HOUSE_RADIUS
        ).astype(int)
    
    df["OwnStonesInHouse"] = sum(
        df[f"stone_{i}_in_house"] * df[f"stone_{i}_is_teams"]
        for i in range(1, 13)
    )
    df["OppStonesInHouse"] = sum(
        df[f"stone_{i}_in_house"] * (1 - df[f"stone_{i}_is_teams"])
        for i in range(1, 13)
    )
    df["StonesInHouseDiff"] = df["OwnStonesInHouse"] - df["OppStonesInHouse"]
    
    return df


def add_closest_stone_features(nnshots):
    """Calculate closest stone distances and counts."""
    nnshots["OwnClosestDist"] = (
        nnshots[[f"stone_{i}_dist_to_button" for i in range(1,13)]]
        .where(nnshots[[f"stone_{i}_is_teams" for i in range(1,13)]].values == 1)
        .min(axis=1)
    )
    nnshots["OwnClosestDist"] = nnshots["OwnClosestDist"].fillna(5000)
    
    nnshots["OppClosestDist"] = (
        nnshots[[f"stone_{i}_dist_to_button" for i in range(1,13)]]
        .where(nnshots[[f"stone_{i}_is_teams" for i in range(1,13)]].values == 0)
        .min(axis=1)
    )
    nnshots["ClosestStoneDiff"] = nnshots["OppClosestDist"] - nnshots["OwnClosestDist"]
    
    nnshots["OwnStonesCloserThanOpp"] = sum(
        ((nnshots[f"stone_{i}_dist_to_button"] < nnshots["OppClosestDist"]) &
         (nnshots[f"stone_{i}_is_teams"] == 1)).astype(int)
        for i in range(1,13))
    
    nnshots["OppStonesCloserThanOwn"] = sum(
        ((nnshots[f"stone_{i}_dist_to_button"] < nnshots["OwnClosestDist"]) &
         (nnshots[f"stone_{i}_is_teams"] == 0)).astype(int)
        for i in range(1,13))
    
    nnshots["StonesCloserDiff"] = nnshots["OwnStonesCloserThanOpp"] - nnshots["OppStonesCloserThanOwn"]
    nnshots["TotalStonesInHouse"] = nnshots["OwnStonesInHouse"] + nnshots["OppStonesInHouse"]
    
    return nnshots


def add_crowding_features(nnshots):
    """Add stone crowding features."""
    nnshots["StonesCrowdedNearButton"] = sum(
        (nnshots[f"stone_{i}_dist_to_button"] < CROWDED_RADIUS).astype(int)
        for i in range(1, 13))
    
    nnshots["StonesMediumCrowdedNearButton"] = sum(
        (nnshots[f"stone_{i}_dist_to_button"] < MEDIUM_CROWDED_RADIUS).astype(int)
        for i in range(1, 13))
    
    return nnshots


def add_power_play_features(nnshots):
    """Add power play usage features."""
    nnshots["GameTeamID"] = (
        nnshots["CompetitionID"].astype(str) + "_" +
        nnshots["SessionID"].astype(str) + "_" +
        nnshots["GameID"].astype(str) + "_" +
        nnshots["TeamID"].astype(str)
    )
    
    pp_used_end = (
        nnshots[nnshots["PowerPlay"] == 1]
        .groupby("GameTeamID")["EndID"]
        .min()
        .rename("PowerPlayUsedEnd")
    )
    nnshots = nnshots.merge(pp_used_end, on="GameTeamID", how="left")
    nnshots["HasUsedPowerPlay"] = (
        (nnshots["PowerPlayUsedEnd"].notna()) &
        (nnshots["EndID"] >= nnshots["PowerPlayUsedEnd"])
    ).astype(int)
    
    return nnshots


def add_reference_team_features(nnshots, games):
    """Add reference team normalization features."""
    nnshots = nnshots.copy()
    
    games["WinningTeamID"] = games.apply(lambda x: x["TeamID2"] if x["Winner"] == 0 else x["TeamID1"], axis=1)
    nnshots = nnshots.merge(games[["CompetitionID", "SessionID", "GameID", "WinningTeamID"]], 
                           on=["CompetitionID", "SessionID", "GameID"], how="left")
    
    nnshots["RefTeamID"] = nnshots.groupby("MatchID")["TeamID"].transform("min")
    nnshots["IsRefTeam"] = (nnshots["TeamID"] == nnshots["RefTeamID"]).astype(int)
    nnshots["RefTeamWon"] = (nnshots["WinningTeamID"] == nnshots["RefTeamID"]).astype(int)
    
    nnshots["RefScoreDiff"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["ScoreDiffStartOfEnd"],
        -nnshots["ScoreDiffStartOfEnd"]
    )
    
    nnshots["RefHasHammer"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["HasHammer"],
        1 - nnshots["HasHammer"]
    )
    
    nnshots["RefShotsRemaining"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["TeamShotsRemainingInEnd"],
        nnshots["OpponentShotsRemainingInEnd"]
    )
    
    nnshots["OppShotsRemaining"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["OpponentShotsRemainingInEnd"],
        nnshots["TeamShotsRemainingInEnd"]
    )
    
    match_teams = (
        nnshots.groupby("MatchID")["TeamID"]
        .unique()
        .to_dict()
    )
    nnshots["OppTeamID"] = nnshots.apply(
        lambda row: (
            match_teams[row["MatchID"]][0]
            if match_teams[row["MatchID"]][1] == row["RefTeamID"]
            else match_teams[row["MatchID"]][1]
        ),
        axis=1
    )
    
    nnshots["RefStonesInHouse"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["OwnStonesInHouse"],
        nnshots["OppStonesInHouse"]
    )
    
    nnshots["OppStonesInHouse"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["OppStonesInHouse"],
        nnshots["OwnStonesInHouse"]
    )
    
    nnshots["RefClosestDist"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["OwnClosestDist"],
        nnshots["OppClosestDist"]
    )
    
    nnshots["OppClosestDist"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["OppClosestDist"],
        nnshots["OwnClosestDist"]
    )
    
    team_pp_map = (
        nnshots.groupby(["MatchID", "TeamID"])["HasUsedPowerPlay"]
        .max()
        .reset_index()
        .set_index(["MatchID", "TeamID"])["HasUsedPowerPlay"]
        .to_dict()
    )
    
    nnshots["RefHasUsedPowerPlay"] = nnshots.apply(
        lambda row: team_pp_map.get((row["MatchID"], row["RefTeamID"]), 0),
        axis=1
    ).astype(int)
    
    nnshots["OppHasUsedPowerPlay"] = nnshots.apply(
        lambda row: team_pp_map.get((row["MatchID"], row["OppTeamID"]), 0),
        axis=1
    ).astype(int)
    
    nnshots["RefStonesCloserDiff"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["StonesCloserDiff"],
        -nnshots["StonesCloserDiff"]
    )
    
    return nnshots


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


def add_game_state_features(nnshots):
    """Add game state features."""
    nnshots["EndsRemaining"] = np.maximum(0, 8 - nnshots["EndID"])
    
    max_end_per_game = nnshots.groupby("MatchID")["EndID"].max().to_dict()
    nnshots["MaxEndInGame"] = nnshots["MatchID"].map(max_end_per_game)
    
    nnshots = nnshots.reset_index(drop=True)
    last_row_per_game = nnshots.groupby("MatchID").tail(1).index
    nnshots["IsLastShotInGame"] = nnshots.index.isin(last_row_per_game).astype(int)
    
    nnshots["EarlyGameEnd"] = (
        (nnshots["IsLastShotInGame"] == 1) & 
        (nnshots["MaxEndInGame"] < 8)
    ).astype(int)
    
    nnshots = nnshots.drop(columns=["MaxEndInGame", "IsLastShotInGame"])
    
    return nnshots


def add_final_state_rows(nnshots, ends_prep):
    """
    Add a synthetic row at the end of each match representing the final game state.
    This helps the model learn that when the game is over (shots=0, ends=0),
    the probability should be 100%/0% based on the final score.
    
    Parameters
    ----------
    nnshots : pd.DataFrame
        Processed shots dataframe
    ends_prep : pd.DataFrame
        Prepared ends dataframe with TotalGameScoreAfterEnd
    
    Returns
    -------
    pd.DataFrame
        nnshots with final state rows appended
    """
    # Get final scores for each team in each game
    final_scores = ends_prep.groupby(['CompetitionID', 'SessionID', 'GameID', 'TeamID'])['TotalGameScoreAfterEnd'].max().reset_index()
    final_scores = final_scores.rename(columns={'TotalGameScoreAfterEnd': 'FinalScore'})
    
    # Get one last shot from each match as a template (RefTeam perspective)
    last_shots = nnshots.groupby('MatchID').tail(1).copy()
    last_shots = last_shots.drop_duplicates(subset=['MatchID'])
    
    final_rows = []
    
    # Create final state row for each match
    for idx, last_shot in last_shots.iterrows():
        match_key = (last_shot['CompetitionID'], last_shot['SessionID'], last_shot['GameID'])
        ref_team = last_shot['RefTeamID']
        opp_team = last_shot['OppTeamID']
        
        # Get final scores for both teams
        ref_final = final_scores[
            (final_scores['CompetitionID'] == last_shot['CompetitionID']) &
            (final_scores['SessionID'] == last_shot['SessionID']) &
            (final_scores['GameID'] == last_shot['GameID']) &
            (final_scores['TeamID'] == ref_team)
        ]['FinalScore'].values
        
        opp_final = final_scores[
            (final_scores['CompetitionID'] == last_shot['CompetitionID']) &
            (final_scores['SessionID'] == last_shot['SessionID']) &
            (final_scores['GameID'] == last_shot['GameID']) &
            (final_scores['TeamID'] == opp_team)
        ]['FinalScore'].values
        
        if len(ref_final) == 0 or len(opp_final) == 0:
            continue
        
        final_row = last_shot.copy()
        final_row['RefShotsRemaining'] = 0
        final_row['OppShotsRemaining'] = 0
        final_row['EndsRemaining'] = 0
        final_row['RefScoreDiff'] = float(ref_final[0] - opp_final[0])
        final_row['RefTeamWon'] = 1 if final_row['RefScoreDiff'] > 0 else 0
        
        # Set stone positions to 0 (game is over, no stones in play)
        for i in range(1, 13):
            final_row[f'stone_{i}_x'] = 0
            final_row[f'stone_{i}_y'] = 0
        
        # Set other features to defaults for game over state
        final_row['RefStonesInHouse'] = 0
        final_row['OppStonesInHouse'] = 0
        final_row['RefClosestDist'] = 4095  # Out of play
        final_row['OppClosestDist'] = 4095
        final_row['RefStonesCloserDiff'] = 0
        final_row['StonesCrowdedNearButton'] = 0
        final_row['StonesMediumCrowdedNearButton'] = 0
        final_row['RefHasHammer'] = 0  # Game is over
        final_row['EndID'] = 9  # Mark as after end 8
        
        final_rows.append(final_row)
    
    if len(final_rows) > 0:
        final_df = pd.DataFrame(final_rows)
        nnshots = pd.concat([nnshots, final_df], ignore_index=True)
    
    return nnshots


def prepare_modeling_data(data_dir="data"):
    """
    Main function to prepare data for modeling.
    
    Returns
    -------
    nnshots : pd.DataFrame
        Processed dataframe ready for modeling
    games : pd.DataFrame
        Games dataframe
    """
    stones, teams, games, ends, competitors, competition = load_data(data_dir)
    
    ends_prep = prepare_ends(ends)
    shots = prepare_shots(stones, ends_prep)
    
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
    
    nnshots = assign_stone_ownership(nnshots)
    nnshots = add_stone_distances(nnshots)
    nnshots = add_stones_in_house(nnshots)
    nnshots = add_closest_stone_features(nnshots)
    nnshots = add_crowding_features(nnshots)
    nnshots = add_power_play_features(nnshots)
    nnshots = add_reference_team_features(nnshots, games)
    nnshots = add_elo_diff_feature(nnshots, games)
    nnshots = add_game_state_features(nnshots)
    nnshots = add_final_state_rows(nnshots, ends_prep)
    
    return nnshots, games

