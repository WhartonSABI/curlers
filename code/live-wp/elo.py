"""
Elo Rating System for Curling Teams

Computes Elo ratings for teams based on game outcomes.
Elo ratings provide a continuous measure of team strength that can be used
as features in predictive models instead of categorical team IDs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_elo_ratings(
    games_df: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    home_advantage: float = 0.0,
    return_history: bool = False
) -> Dict[int, float]:
    """
    Compute Elo ratings for teams from game results.
    
    Parameters
    ----------
    games_df : pd.DataFrame
        DataFrame containing game results with columns:
        - CompetitionID, SessionID, GameID (for ordering games chronologically)
        - TeamID1, TeamID2: IDs of the two teams
        - Winner: 0 if TeamID2 won, 1 if TeamID1 won
    initial_rating : float, default=1500.0
        Starting Elo rating for all teams
    k_factor : float, default=32.0
        K-factor determines how much ratings change after each game.
        Higher values = more volatile ratings. Standard is 32.
    home_advantage : float, default=0.0
        Home advantage adjustment (in Elo points). Positive means TeamID1 has advantage.
        For curling, this is typically 0 since there's no home advantage.
    return_history : bool, default=False
        If True, also return rating history for each team
    
    Returns
    -------
    Dict[int, float]
        Dictionary mapping TeamID to Elo rating
    """
    # Initialize all teams with initial rating
    elo_ratings: Dict[int, float] = {}
    rating_history: Dict[int, list] = {} if return_history else {}
    
    # Sort games chronologically (by competition, session, game)
    games_sorted = games_df.sort_values(
        ['CompetitionID', 'SessionID', 'GameID']
    ).copy()
    
    # Process each game
    for _, game in games_sorted.iterrows():
        team1 = int(game['TeamID1'])
        team2 = int(game['TeamID2'])
        winner = int(game['TeamID1']) if game['Winner'] == 1 else int(game['TeamID2'])
        
        # Get current ratings (initialize if first time seeing team)
        r1 = elo_ratings.get(team1, initial_rating)
        r2 = elo_ratings.get(team2, initial_rating)
        
        # Apply home advantage (if any)
        r1_adjusted = r1 + home_advantage
        
        # Expected scores (probability of winning)
        e1 = 1 / (1 + 10 ** ((r2 - r1_adjusted) / 400))
        e2 = 1 - e1
        
        # Actual scores (1 for winner, 0 for loser)
        s1 = 1 if winner == team1 else 0
        s2 = 1 if winner == team2 else 0
        
        # Update ratings
        elo_ratings[team1] = r1 + k_factor * (s1 - e1)
        elo_ratings[team2] = r2 + k_factor * (s2 - e2)
        
        # Store history if requested
        if return_history:
            if team1 not in rating_history:
                rating_history[team1] = []
            if team2 not in rating_history:
                rating_history[team2] = []
            rating_history[team1].append(elo_ratings[team1])
            rating_history[team2].append(elo_ratings[team2])
    
    if return_history:
        return elo_ratings, rating_history
    return elo_ratings


def compute_elo_ratings_by_competition(
    games_df: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    carry_over: bool = True
) -> Dict[int, Dict[int, float]]:
    """
    Compute Elo ratings separately for each competition, optionally carrying
    over ratings from previous competitions.
    
    Parameters
    ----------
    games_df : pd.DataFrame
        DataFrame containing game results
    initial_rating : float, default=1500.0
        Starting Elo rating for teams in first competition
    k_factor : float, default=32.0
        K-factor for rating updates
    carry_over : bool, default=True
        If True, teams keep their final rating from previous competition.
        If False, ratings reset to initial_rating for each competition.
    
    Returns
    -------
    Dict[int, Dict[int, float]]
        Nested dictionary: {CompetitionID: {TeamID: EloRating}}
    """
    competition_ratings = {}
    global_ratings = {}
    
    for comp_id in sorted(games_df['CompetitionID'].unique()):
        comp_games = games_df[games_df['CompetitionID'] == comp_id].copy()
        
        # Initialize ratings for this competition
        comp_ratings = {}
        for team_id in set(comp_games['TeamID1'].unique()) | set(comp_games['TeamID2'].unique()):
            if carry_over and team_id in global_ratings:
                comp_ratings[team_id] = global_ratings[team_id]
            else:
                comp_ratings[team_id] = initial_rating
        
        # Process games in this competition
        comp_games_sorted = comp_games.sort_values(['SessionID', 'GameID'])
        
        for _, game in comp_games_sorted.iterrows():
            team1 = int(game['TeamID1'])
            team2 = int(game['TeamID2'])
            winner = int(game['TeamID1']) if game['Winner'] == 1 else int(game['TeamID2'])
            
            r1 = comp_ratings[team1]
            r2 = comp_ratings[team2]
            
            e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            s1 = 1 if winner == team1 else 0
            
            comp_ratings[team1] = r1 + k_factor * (s1 - e1)
            comp_ratings[team2] = r2 + k_factor * ((1 - s1) - (1 - e1))
        
        # Update global ratings
        global_ratings.update(comp_ratings)
        competition_ratings[comp_id] = comp_ratings.copy()
    
    return competition_ratings


def add_elo_features_to_shots(
    shots_df: pd.DataFrame,
    games_df: pd.DataFrame,
    elo_ratings: Optional[Dict[int, float]] = None,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0
) -> pd.DataFrame:
    """
    Add Elo rating features to shots dataframe.
    
    Parameters
    ----------
    shots_df : pd.DataFrame
        Shots dataframe with columns: CompetitionID, SessionID, GameID, TeamID
    games_df : pd.DataFrame
        Games dataframe with game results
    elo_ratings : Dict[int, float], optional
        Pre-computed Elo ratings. If None, will compute from games_df
    initial_rating : float, default=1500.0
        Initial rating if computing Elo ratings
    k_factor : float, default=32.0
        K-factor if computing Elo ratings
    
    Returns
    -------
    pd.DataFrame
        Shots dataframe with added columns:
        - TeamElo: Elo rating of the team in this row
        - OppElo: Elo rating of the opponent team
        - EloDiff: TeamElo - OppElo (positive = stronger team)
    """
    shots = shots_df.copy()
    
    # Compute Elo ratings if not provided
    if elo_ratings is None:
        elo_ratings = compute_elo_ratings(
            games_df, 
            initial_rating=initial_rating, 
            k_factor=k_factor
        )
    
    # Get opponent team for each row
    # First, merge with games to get TeamID1 and TeamID2
    games_subset = games_df[['CompetitionID', 'SessionID', 'GameID', 'TeamID1', 'TeamID2']].copy()
    shots = shots.merge(
        games_subset,
        on=['CompetitionID', 'SessionID', 'GameID'],
        how='left'
    )
    
    # Determine opponent team
    shots['OppTeamID'] = shots.apply(
        lambda row: row['TeamID2'] if row['TeamID'] == row['TeamID1'] else row['TeamID1'],
        axis=1
    )
    
    # Add Elo ratings
    shots['TeamElo'] = shots['TeamID'].map(elo_ratings).fillna(initial_rating)
    shots['OppElo'] = shots['OppTeamID'].map(elo_ratings).fillna(initial_rating)
    shots['EloDiff'] = shots['TeamElo'] - shots['OppElo']
    
    # Clean up temporary columns
    shots = shots.drop(columns=['TeamID1', 'TeamID2', 'OppTeamID'])
    
    return shots


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load data
    data_dir = "data"
    if not os.path.exists(data_dir):
        # Try going up one level
        data_dir = "../data"
    
    games = pd.read_csv(f"{data_dir}/Games.csv", low_memory=False)
    
    # Compute Elo ratings
    elo_ratings = compute_elo_ratings(games)
    
    # Print top 10 teams by Elo rating
    sorted_teams = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 Teams by Elo Rating:")
    print("-" * 40)
    for team_id, rating in sorted_teams[:10]:
        print(f"Team {team_id}: {rating:.1f}")
    
    # Save ratings to CSV
    elo_df = pd.DataFrame([
        {'TeamID': team_id, 'EloRating': rating}
        for team_id, rating in elo_ratings.items()
    ])
    elo_df.to_csv(f"{data_dir}/elo_ratings.csv", index=False)
    print(f"\nElo ratings saved to {data_dir}/elo_ratings.csv")

