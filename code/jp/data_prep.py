"""
Data preparation and feature engineering for curling shot-level data.
"""

import pandas as pd
import numpy as np
import os


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
    
    # CRITICAL: Sort by ShotID before computing shot numbers to ensure correct order
    # Without this, any upstream merge/order change can scramble shot numbers
    shots = shots.sort_values(["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]).reset_index(drop=True)
    
    shots["TotalShotNumberInEnd"] = shots.groupby(["CompetitionID", "SessionID", "GameID", "EndID"]).cumcount()+1
    shots["TotalShotsRemainingInEnd"] = 10 - shots["TotalShotNumberInEnd"]
    shots["PowerPlay"] = shots["PowerPlay"].fillna(0)
    
    shots["HasHammer"] = shots["TotalShotNumberInEnd"].apply(lambda x: 1 if x in [2,4,6,8,10] else 0)
    
    shots["TeamPlayerID"] = (shots["TeamID"].astype(int).astype(str) 
                             + shots["PlayerID"].astype(int).astype(str)).astype(int)
    # Use separator to avoid collisions (e.g., (1,11,1) vs (11,1,1))
    shots["MatchID"] = (
        shots["CompetitionID"].astype(str) + "_" +
        shots["SessionID"].astype(str) + "_" +
        shots["GameID"].astype(str)
    )
    
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
    """
    Add power play usage features.
    
    PowerPlay = 1 means positioned stones moved right
    PowerPlay = 2 means positioned stones moved left
    PowerPlay in {1, 2} means power play was used.
    
    Fix 1: Treat PowerPlay in {1,2} as "used" (not just > 0)
    Fix 2: PP availability must be "before this end" (EndID < PowerPlayUsedEnd, strict)
    """
    nnshots = nnshots.copy()
    nnshots["PowerPlay"] = nnshots["PowerPlay"].fillna(0).astype(int)
    nnshots["PowerPlayUsedThisEnd"] = nnshots["PowerPlay"].isin([1, 2]).astype(int)
    
    nnshots["GameTeamID"] = (
        nnshots["CompetitionID"].astype(str) + "_" +
        nnshots["SessionID"].astype(str) + "_" +
        nnshots["GameID"].astype(str) + "_" +
        nnshots["TeamID"].astype(str)
    )
    
    # Find the first end where each team used power play
    pp_used_end = (
        nnshots[nnshots["PowerPlayUsedThisEnd"] == 1]
        .groupby("GameTeamID")["EndID"]
        .min()
        .rename("PowerPlayUsedEnd")
    )
    nnshots = nnshots.merge(pp_used_end, on="GameTeamID", how="left")
    
    # Decision-time availability at start of end
    # PP is available if it has NOT been used in any prior end (strict <)
    nnshots["PowerPlayAvailableBeforeEnd"] = (
        nnshots["PowerPlayUsedEnd"].isna() | (nnshots["EndID"] < nnshots["PowerPlayUsedEnd"])
    ).astype(int)
    
    # Convenience: HasUsedPowerPlayBeforeEnd = 1 - PowerPlayAvailableBeforeEnd
    nnshots["HasUsedPowerPlayBeforeEnd"] = (1 - nnshots["PowerPlayAvailableBeforeEnd"]).astype(int)
    
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
    
    # RefPowerPlay: 1 if ref team used power play in this end, 0 otherwise
    # PowerPlay is per-shot, so we need to check if ref team used it in the end
    # Fix 1: Treat PowerPlay in {1,2} as used (not just > 0)
    ref_pp_by_end = (
        nnshots[nnshots["IsRefTeam"] == 1]
        .groupby(["CompetitionID", "SessionID", "GameID", "EndID"])["PowerPlayUsedThisEnd"]
        .max()
        .reset_index()
        .rename(columns={"PowerPlayUsedThisEnd": "RefPowerPlayInEnd"})
    )
    nnshots = nnshots.merge(
        ref_pp_by_end,
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="left"
    )
    # Convert to binary: 1 if PowerPlayUsedThisEnd == 1, 0 otherwise
    nnshots["RefPowerPlay"] = (nnshots["RefPowerPlayInEnd"].fillna(0).astype(int) > 0).astype(int)
    
    # OppPowerPlay: 1 if opponent used power play in this end, 0 otherwise
    opp_pp_by_end = (
        nnshots[nnshots["IsRefTeam"] == 0]
        .groupby(["CompetitionID", "SessionID", "GameID", "EndID"])["PowerPlayUsedThisEnd"]
        .max()
        .reset_index()
        .rename(columns={"PowerPlayUsedThisEnd": "OppPowerPlayInEnd"})
    )
    nnshots = nnshots.merge(
        opp_pp_by_end,
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="left"
    )
    # Convert to binary: 1 if PowerPlayUsedThisEnd == 1, 0 otherwise
    nnshots["OppPowerPlay"] = (nnshots["OppPowerPlayInEnd"].fillna(0).astype(int) > 0).astype(int)
    
    nnshots = nnshots.drop(columns=["RefPowerPlayInEnd", "OppPowerPlayInEnd"])
    
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
    
    # Create time-varying PP status per team per end (not constant per match)
    # Get unique (MatchID, TeamID, EndID) combinations with PP status
    pp_status = nnshots[
        ["MatchID", "TeamID", "EndID", "HasUsedPowerPlayBeforeEnd", "PowerPlayAvailableBeforeEnd"]
    ].drop_duplicates()
    
    # Create ref team perspective PP status
    # Match TeamID to RefTeamID for each row
    ref_pp = pp_status.copy()
    ref_pp = ref_pp.rename(columns={
        "TeamID": "RefTeamID",
        "HasUsedPowerPlayBeforeEnd": "RefHasUsedPowerPlay",
        "PowerPlayAvailableBeforeEnd": "RefPowerPlayAvailable"
    })
    ref_pp = ref_pp[["MatchID", "RefTeamID", "EndID", "RefHasUsedPowerPlay", "RefPowerPlayAvailable"]]
    
    # Create opp team perspective PP status
    # Match TeamID to OppTeamID for each row
    opp_pp = pp_status.copy()
    opp_pp = opp_pp.rename(columns={
        "TeamID": "OppTeamID",
        "HasUsedPowerPlayBeforeEnd": "OppHasUsedPowerPlay",
        "PowerPlayAvailableBeforeEnd": "OppPowerPlayAvailable"
    })
    opp_pp = opp_pp[["MatchID", "OppTeamID", "EndID", "OppHasUsedPowerPlay", "OppPowerPlayAvailable"]]
    
    # Merge ref and opp PP status by end (time-varying, not constant per match)
    nnshots = nnshots.merge(
        ref_pp,
        on=["MatchID", "RefTeamID", "EndID"],
        how="left"
    )
    nnshots = nnshots.merge(
        opp_pp,
        on=["MatchID", "OppTeamID", "EndID"],
        how="left"
    )
    
    # Fill any missing values (shouldn't happen, but safety)
    nnshots["RefHasUsedPowerPlay"] = nnshots["RefHasUsedPowerPlay"].fillna(0).astype(int)
    nnshots["OppHasUsedPowerPlay"] = nnshots["OppHasUsedPowerPlay"].fillna(0).astype(int)
    nnshots["RefPowerPlayAvailable"] = nnshots["RefPowerPlayAvailable"].fillna(1).astype(int)
    nnshots["OppPowerPlayAvailable"] = nnshots["OppPowerPlayAvailable"].fillna(1).astype(int)
    
    nnshots["RefStonesCloserDiff"] = np.where(
        nnshots["IsRefTeam"] == 1,
        nnshots["StonesCloserDiff"],
        -nnshots["StonesCloserDiff"]
    )
    
    return nnshots




def add_hammer_team_features(nnshots):
    """
    Add features from hammer team's perspective for EP model.
    
    EP model should always be from the perspective of the team with the hammer,
    since power play is only available to the hammer team.
    
    Creates:
    - HamShotsRemaining, OppShotsRemaining
    - HamStonesCloserDiff
    - HamClosestDist, OppClosestDist
    - HamStonesInHouse, OppStonesInHouse
    - PowerPlay (1 if hammer team using PP this end)
    - HamScoreDiff (score diff from hammer team's perspective)
    - HamEloDiff (ELO diff from hammer team's perspective)
    """
    nnshots = nnshots.copy()
    
    # Hammer team perspective: flip when ref team doesn't have hammer
    nnshots["HamShotsRemaining"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["RefShotsRemaining"],
        nnshots["OppShotsRemaining"]
    )
    
    nnshots["OppShotsRemainingHam"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["OppShotsRemaining"],
        nnshots["RefShotsRemaining"]
    )
    
    nnshots["HamStonesCloserDiff"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["RefStonesCloserDiff"],
        -nnshots["RefStonesCloserDiff"]
    )
    
    nnshots["HamClosestDist"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["RefClosestDist"],
        nnshots["OppClosestDist"]
    )
    
    nnshots["OppClosestDistHam"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["OppClosestDist"],
        nnshots["RefClosestDist"]
    )
    
    nnshots["HamStonesInHouse"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["RefStonesInHouse"],
        nnshots["OppStonesInHouse"]
    )
    
    nnshots["OppStonesInHouseHam"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["OppStonesInHouse"],
        nnshots["RefStonesInHouse"]
    )
    
    # PowerPlay: 1 if hammer team using PP this end
    nnshots["PowerPlay"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["RefPowerPlay"],
        nnshots["OppPowerPlay"]
    )
    
    # Score diff from hammer team's perspective
    nnshots["HamScoreDiff"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["RefScoreDiff"],
        -nnshots["RefScoreDiff"]
    )
    
    # ELO diff from hammer team's perspective
    nnshots["HamEloDiff"] = np.where(
        nnshots["RefHasHammer"] == 1,
        nnshots["RefEloDiff"],
        -nnshots["RefEloDiff"]
    )
    
    return nnshots


def add_game_state_features(nnshots):
    """Add game state features."""
    # Compute max end per game (handles early concessions and extra ends)
    max_end_per_game = nnshots.groupby("MatchID")["EndID"].max().to_dict()
    nnshots["MaxEndInGame"] = nnshots["MatchID"].map(max_end_per_game)
    
    # EndsRemaining: ends remaining AFTER current end (excluding current end)
    nnshots["EndsRemaining"] = np.maximum(0, nnshots["MaxEndInGame"] - nnshots["EndID"])
    
    nnshots = nnshots.reset_index(drop=True)
    last_row_per_game = nnshots.groupby("MatchID").tail(1).index
    nnshots["IsLastShotInGame"] = nnshots.index.isin(last_row_per_game).astype(int)
    
    nnshots["EarlyGameEnd"] = (
        (nnshots["IsLastShotInGame"] == 1) & 
        (nnshots["MaxEndInGame"] < 8)
    ).astype(int)
    
    nnshots = nnshots.drop(columns=["IsLastShotInGame"])
    
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
        # Set EndID to MaxEndInGame + 1 (handles early concessions and extra ends)
        if 'MaxEndInGame' in last_shot.index:
            max_end = last_shot['MaxEndInGame']
        else:
            # Fallback: compute from nnshots if not in last_shot
            max_end = nnshots[nnshots['MatchID'] == last_shot['MatchID']]['EndID'].max() if len(nnshots) > 0 else 8
        final_row['EndID'] = max_end + 1
        
        final_rows.append(final_row)
    
    if len(final_rows) > 0:
        final_df = pd.DataFrame(final_rows)
        nnshots = pd.concat([nnshots, final_df], ignore_index=True)
    
    return nnshots




def build_start_of_end_df(ends_prep, stones, games=None):
    """
    Build end-level dataframe with one row per match-end at the start of the end.
    
    This is the correct decision-time state for PP decisions, which happen
    at the beginning of an end, before any stones are thrown.
    
    Parameters
    ----------
    ends_prep : pd.DataFrame
        Prepared ends dataframe with score and result columns
    stones : pd.DataFrame
        Raw stones dataframe (for hammer inference)
    games : pd.DataFrame, optional
        Games dataframe (for team identification)
    
    Returns
    -------
    pd.DataFrame
        End-level dataframe with columns:
        - CompetitionID, SessionID, GameID, EndID
        - RefTeamID, OppTeamID
        - RefScoreDiffStartOfEnd (score diff from ref perspective at start)
        - RefHasHammerStartOfEnd (1 if ref has hammer, 0 otherwise)
        - RefEndDifferential (end result from ref perspective: Result - OpponentResult)
        - PPUsedThisEnd (1 if ref used PP, -1 if opp used PP, 0 if neither)
        - RefPPAvailableBeforeEnd (1 if ref PP available at start, 0 otherwise)
        - OppPPAvailableBeforeEnd (1 if opp PP available at start, 0 otherwise)
        - RefEloDiff (optional, if games provided)
    """
    # Identify ref and opp teams per match
    # Use min TeamID as ref team (consistent with existing code)
    match_teams = (
        ends_prep.groupby(["CompetitionID", "SessionID", "GameID"])["TeamID"]
        .unique()
        .reset_index()
    )
    match_teams["RefTeamID"] = match_teams["TeamID"].apply(lambda x: min(x))
    match_teams["OppTeamID"] = match_teams["TeamID"].apply(lambda x: max(x))
    match_teams = match_teams.drop(columns=["TeamID"])
    
    # Merge to get ref/opp team IDs
    ends_with_teams = ends_prep.merge(
        match_teams,
        on=["CompetitionID", "SessionID", "GameID"],
        how="left"
    )
    
    # Create one row per end from ref perspective
    ref_ends = ends_with_teams[ends_with_teams["TeamID"] == ends_with_teams["RefTeamID"]].copy()
    opp_ends = ends_with_teams[ends_with_teams["TeamID"] == ends_with_teams["OppTeamID"]].copy()
    
    # Merge ref and opp data
    end_df = ref_ends.merge(
        opp_ends[["CompetitionID", "SessionID", "GameID", "EndID", "Result", "TotalGameScoreStartOfEnd"]],
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        suffixes=("_ref", "_opp")
    )
    
    # Compute ref perspective columns
    end_df["RefScoreDiffStartOfEnd"] = (
        end_df["TotalGameScoreStartOfEnd_ref"] - end_df["TotalGameScoreStartOfEnd_opp"]
    )
    end_df["RefEndDifferential"] = end_df["Result_ref"] - end_df["Result_opp"]
    
    # Infer hammer team from stones: last shot in end has hammer
    # Sort stones by ShotID to get correct order
    stones_sorted = stones.sort_values(
        ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
    )
    hammer_team = (
        stones_sorted.groupby(["CompetitionID", "SessionID", "GameID", "EndID"])["TeamID"]
        .last()
        .reset_index()
        .rename(columns={"TeamID": "HammerTeamID"})
    )
    
    end_df = end_df.merge(
        hammer_team,
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="left"
    )
    end_df["RefHasHammerStartOfEnd"] = (
        end_df["HammerTeamID"] == end_df["RefTeamID"]
    ).astype(int)
    
    # PP usage this end: PowerPlay in {1, 2} means used
    # Create single variable: 1 if ref uses PP, -1 if opp uses PP, 0 if neither
    end_df["RefUsesPPThisEnd"] = (
        end_df["PowerPlay"].fillna(0).astype(int).isin([1, 2])
    ).astype(int)
    
    # For opp PP, we need to check opp team's PowerPlay
    # Merge opp ends to get opp PowerPlay
    opp_pp = opp_ends[["CompetitionID", "SessionID", "GameID", "EndID", "PowerPlay"]].copy()
    opp_pp = opp_pp.rename(columns={"PowerPlay": "OppPowerPlay"})
    end_df = end_df.merge(
        opp_pp,
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="left"
    )
    end_df["OppUsesPPThisEnd"] = (
        end_df["OppPowerPlay"].fillna(0).astype(int).isin([1, 2])
    ).astype(int)
    
    # Create single PP variable: 1 if ref uses, -1 if opp uses, 0 if neither
    end_df["PPUsedThisEnd"] = end_df["RefUsesPPThisEnd"] - end_df["OppUsesPPThisEnd"]
    
    # Compute PP availability before end
    # Create GameTeamID for tracking
    end_df["RefGameTeamID"] = (
        end_df["CompetitionID"].astype(str) + "_" +
        end_df["SessionID"].astype(str) + "_" +
        end_df["GameID"].astype(str) + "_" +
        end_df["RefTeamID"].astype(str)
    )
    end_df["OppGameTeamID"] = (
        end_df["CompetitionID"].astype(str) + "_" +
        end_df["SessionID"].astype(str) + "_" +
        end_df["GameID"].astype(str) + "_" +
        end_df["OppTeamID"].astype(str)
    )
    
    # Find first end where each team used PP
    ref_pp_used_end = (
        end_df[end_df["RefUsesPPThisEnd"] == 1]
        .groupby("RefGameTeamID")["EndID"]
        .min()
        .rename("RefPPUsedEnd")
    )
    opp_pp_used_end = (
        end_df[end_df["OppUsesPPThisEnd"] == 1]
        .groupby("OppGameTeamID")["EndID"]
        .min()
        .rename("OppPPUsedEnd")
    )
    
    end_df = end_df.merge(
        ref_pp_used_end,
        left_on="RefGameTeamID",
        right_index=True,
        how="left"
    )
    end_df = end_df.merge(
        opp_pp_used_end,
        left_on="OppGameTeamID",
        right_index=True,
        how="left"
    )
    
    # PP available if not used yet (strict < check)
    end_df["RefPPAvailableBeforeEnd"] = (
        end_df["RefPPUsedEnd"].isna() | (end_df["EndID"] < end_df["RefPPUsedEnd"])
    ).astype(int)
    end_df["OppPPAvailableBeforeEnd"] = (
        end_df["OppPPUsedEnd"].isna() | (end_df["EndID"] < end_df["OppPPUsedEnd"])
    ).astype(int)
    
    # Add Elo diff if games provided
    if games is not None:
        from elo import compute_elo_ratings
        elo_ratings = compute_elo_ratings(games)
        end_df["RefTeamElo"] = end_df["RefTeamID"].map(elo_ratings).fillna(1500.0)
        end_df["OppTeamElo"] = end_df["OppTeamID"].map(elo_ratings).fillna(1500.0)
        end_df["RefEloDiff"] = end_df["RefTeamElo"] - end_df["OppTeamElo"]
        end_df = end_df.drop(columns=["RefTeamElo", "OppTeamElo"])
    
    # Add early quit indicator: game ends before end 8
    max_end_per_game = end_df.groupby(["CompetitionID", "SessionID", "GameID"])["EndID"].max().reset_index()
    max_end_per_game = max_end_per_game.rename(columns={"EndID": "MaxEndInGame"})
    end_df = end_df.merge(max_end_per_game, on=["CompetitionID", "SessionID", "GameID"], how="left")
    
    # Early quit: game ends before end 8 (concession)
    # For each end, check if it's the last end and if MaxEndInGame < 8
    end_df["IsLastEndInGame"] = (end_df["EndID"] == end_df["MaxEndInGame"]).astype(int)
    end_df["EarlyQuit"] = ((end_df["IsLastEndInGame"] == 1) & (end_df["MaxEndInGame"] < 8)).astype(int)
    
    # Select and rename columns for final output
    output_cols = [
        "CompetitionID", "SessionID", "GameID", "EndID",
        "RefTeamID", "OppTeamID",
        "RefScoreDiffStartOfEnd",
        "RefHasHammerStartOfEnd",
        "RefEndDifferential",
        "PPUsedThisEnd",  # Single variable: 1 if ref uses, -1 if opp uses, 0 if neither
        "RefPPAvailableBeforeEnd",
        "OppPPAvailableBeforeEnd",
        "EarlyQuit",  # 1 if game ends after this end (early concession)
        "MaxEndInGame"  # Maximum end in this game
    ]
    if games is not None and "RefEloDiff" in end_df.columns:
        output_cols.append("RefEloDiff")
    
    end_df = end_df[output_cols].copy()
    
    return end_df

