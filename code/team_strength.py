"""
Regularized Bradley-Terry team strength model for curling teams.

The fitted ability values are log-odds scale Bradley-Terry parameters. For a
game between teams i and j, the model estimates:

    P(i beats j) = logistic(ability_i - ability_j)

No point-scale rescaling is applied.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def compute_bt_ratings(
    games_df: pd.DataFrame,
    ridge_alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    return_model: bool = False,
) -> Dict[int, float]:
    """
    Fit ridge-regularized Bradley-Terry abilities from game outcomes.

    Parameters
    ----------
    games_df : pd.DataFrame
        DataFrame containing game results with columns:
        - TeamID1, TeamID2: IDs of the two teams
        - Winner: 0 if TeamID2 won, 1 if TeamID1 won
    ridge_alpha : float, default=1.0
        L2 regularization strength. Larger values shrink abilities more strongly
        toward 0.
    max_iter : int, default=1000
        Maximum optimizer iterations for the logistic fit.
    tol : float, default=1e-8
        Optimizer convergence tolerance.
    return_model : bool, default=False
        If True, return ``(abilities, model)``.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping TeamID to centered Bradley-Terry ability.
    """
    if ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive for ridge Bradley-Terry")

    required_cols = {"TeamID1", "TeamID2", "Winner"}
    missing_cols = required_cols - set(games_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required game columns: {sorted(missing_cols)}")

    games = games_df.dropna(subset=["TeamID1", "TeamID2", "Winner"]).copy()
    if games.empty:
        return ({}, None) if return_model else {}

    teams = pd.Index(
        pd.concat([games["TeamID1"], games["TeamID2"]], ignore_index=True)
        .astype(int)
        .unique()
    ).sort_values()
    team_to_idx = {int(team_id): idx for idx, team_id in enumerate(teams)}

    n_games = len(games)
    team1_idx = games["TeamID1"].astype(int).map(team_to_idx).to_numpy()
    team2_idx = games["TeamID2"].astype(int).map(team_to_idx).to_numpy()

    row_idx = np.repeat(np.arange(n_games), 2)
    col_idx = np.empty(2 * n_games, dtype=int)
    col_idx[0::2] = team1_idx
    col_idx[1::2] = team2_idx
    data = np.tile([1.0, -1.0], n_games)

    X = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_games, len(teams)))
    y = games["Winner"].astype(int).to_numpy()

    model = LogisticRegression(
        C=1.0 / ridge_alpha,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=max_iter,
        tol=tol,
    )
    model.fit(X, y)

    abilities = model.coef_[0].astype(float)
    abilities = abilities - abilities.mean()
    bt_ratings = {int(team_id): float(ability) for team_id, ability in zip(teams, abilities)}

    if return_model:
        return bt_ratings, model
    return bt_ratings


def bt_win_probability(ability_diff: float) -> float:
    """Convert a Bradley-Terry ability difference to win probability."""
    return float(expit(ability_diff))


def add_bt_features_to_shots(
    shots_df: pd.DataFrame,
    games_df: pd.DataFrame,
    bt_ratings: Optional[Dict[int, float]] = None,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Add Bradley-Terry ability features to a shots dataframe.

    Returns columns:
    - TeamBTAbility
    - OppBTAbility
    - BTAbilityDiff
    """
    shots = shots_df.copy()

    if bt_ratings is None:
        bt_ratings = compute_bt_ratings(games_df, ridge_alpha=ridge_alpha)

    games_subset = games_df[
        ["CompetitionID", "SessionID", "GameID", "TeamID1", "TeamID2"]
    ].copy()
    shots = shots.merge(
        games_subset,
        on=["CompetitionID", "SessionID", "GameID"],
        how="left",
    )

    shots["OppTeamID"] = shots.apply(
        lambda row: row["TeamID2"] if row["TeamID"] == row["TeamID1"] else row["TeamID1"],
        axis=1,
    )

    shots["TeamBTAbility"] = shots["TeamID"].map(bt_ratings).fillna(0.0)
    shots["OppBTAbility"] = shots["OppTeamID"].map(bt_ratings).fillna(0.0)
    shots["BTAbilityDiff"] = shots["TeamBTAbility"] - shots["OppBTAbility"]

    shots = shots.drop(columns=["TeamID1", "TeamID2", "OppTeamID"])

    return shots


if __name__ == "__main__":
    import os

    data_dir = "data"
    if not os.path.exists(data_dir):
        data_dir = "../data"

    games = pd.read_csv(f"{data_dir}/Games.csv", low_memory=False)
    bt_ratings = compute_bt_ratings(games)

    sorted_teams = sorted(bt_ratings.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 Teams by Bradley-Terry Ability:")
    print("-" * 40)
    for team_id, ability in sorted_teams[:10]:
        print(f"Team {team_id}: {ability:.4f}")

    bt_df = pd.DataFrame(
        [
            {"TeamID": team_id, "BTAbility": ability}
            for team_id, ability in bt_ratings.items()
        ]
    )
    bt_df.to_csv(f"{data_dir}/bt_ratings.csv", index=False)
    print(f"\nBradley-Terry ratings saved to {data_dir}/bt_ratings.csv")
