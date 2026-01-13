"""
Decision analysis utilities for curling models.

This module provides helper functions for decision analysis, but the main
PP policy computation is now in ep_policy.py using optimal stopping DP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_decisions(df, value_col="DeltaWP"):
    """
    Aggregate decisions by EndID and ScoreDiff.
    
    Parameters
    ----------
    df : pd.DataFrame
        Decision dataframe with EndID, ScoreDiff, and value column
    value_col : str, default="DeltaWP"
        Column name to aggregate
    
    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with mean and count
    """
    return df.groupby(["EndID", "ScoreDiff"]).agg(
        mean_value=(value_col, "mean"),
        count=(value_col, "size")
    ).reset_index()

