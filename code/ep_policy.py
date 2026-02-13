"""
Optimal Power Play timing policy using dynamic programming.

This module implements a finite-horizon DP to determine optimal PP deployment
as a function of (end, score diff, hammer, PP availability).

The DP optimizes expected win probability by considering:
- Current end (1-8)
- Current score differential (ref - opp)
- Who has hammer this end (ref=1, opp=0)
- PP availability for both teams (ref_pp_avail, opp_pp_avail)

State transitions follow mixed doubles rules:
- If ref scores (d > 0): ref LOSES hammer, opp gets it
- If opp scores (d < 0): opp LOSES hammer, ref gets it
- If blank (d == 0): hammer switches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from ep_end import predict_end_differential_distribution, predict_early_quit_probability


def extra_end_value(score_diff, hammer, ref_pp_avail, opp_pp_avail, 
                   extra_end_ep_model, extra_end_differential_classes, extra_end_class_to_diff,
                   value_cache, elo_diff=0.0, max_extra_ends=5):
    """
    Compute value for extra end continuation (infinite horizon approximation).
    
    Extra ends continue until someone scores. We approximate this by:
    1. If someone scores (score_diff != 0), game ends
    2. If blank (score_diff == 0), continue to next extra end
    3. Cap at max_extra_ends to prevent infinite recursion
    
    Parameters
    ----------
    score_diff : int
        Current score differential (ref - opp)
    hammer : int
        1 if ref has hammer, 0 if opp has hammer
    ref_pp_avail : int
        1 if ref PP available, 0 if not
    opp_pp_avail : int
        1 if opp PP available, 0 if not
    extra_end_ep_model : XGBClassifier
        Trained EP model for extra ends
    extra_end_differential_classes : np.ndarray
        Array of possible differential values for extra ends
    extra_end_class_to_diff : dict
        Mapping from class index to differential value
    value_cache : dict
        Memoization cache
    elo_diff : float, default=0.0
        Elo difference (ref - opp)
    max_extra_ends : int, default=5
        Maximum number of extra ends to simulate
    
    Returns
    -------
    float
        Expected win probability
    """
    # If someone scored, game ends
    if score_diff > 0:
        return 1.0
    elif score_diff < 0:
        return 0.0
    
    # If we've simulated too many extra ends, use Elo-based tie-breaker
    # (This shouldn't happen often, but prevents infinite recursion)
    if max_extra_ends <= 0:
        if elo_diff == 0.0:
            return 0.5
        else:
            return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    
    # Cache key for extra ends
    elo_bucket = round(elo_diff / 50.0) * 50.0
    cache_key = ("extra", score_diff, hammer, ref_pp_avail, opp_pp_avail, elo_bucket, max_extra_ends)
    if cache_key in value_cache:
        return value_cache[cache_key]
    
    # Create state row for extra end EP prediction
    state_row = pd.Series({
        "EndsRemaining": 0,  # Extra ends have 0 ends remaining
        "RefHasHammerStartOfEnd": hammer,
        "RefScoreDiffStartOfEnd": score_diff,
        "PPUsedThisEnd": 0,
        "RefPPAvailableBeforeEnd": ref_pp_avail,
        "OppPPAvailableBeforeEnd": opp_pp_avail,
        "RefEloDiff": elo_diff
    })
    
    # Power Play cannot be used in extra ends
    actions = [0]
    
    best_value = -np.inf
    
    for use_pp in actions:
        state_row["PPUsedThisEnd"] = 0
        
        # Predict extra end differential distribution
        prob_dist = predict_end_differential_distribution(
            state_row, extra_end_ep_model, extra_end_differential_classes, 
            extra_end_class_to_diff, is_extra_end=True
        )
        
        expected_value = 0.0
        for end_diff, prob in prob_dist.items():
            if prob > 0:
                next_score = score_diff + end_diff
                next_hammer_val = next_hammer(hammer, end_diff)
                
                if hammer == 1:
                    next_ref_pp = next_pp_availability(ref_pp_avail, use_pp == 1)
                    next_opp_pp = opp_pp_avail
                else:
                    next_ref_pp = ref_pp_avail
                    next_opp_pp = next_pp_availability(opp_pp_avail, use_pp == 1)
                
                # Recursive call for next extra end
                next_value = extra_end_value(
                    next_score, next_hammer_val, next_ref_pp, next_opp_pp,
                    extra_end_ep_model, extra_end_differential_classes, extra_end_class_to_diff,
                    value_cache, elo_diff, max_extra_ends - 1
                )
                
                expected_value += prob * next_value
        
        if hammer == 0 and opp_pp_avail == 1:
            if expected_value < best_value or best_value == -np.inf:
                best_value = expected_value
        else:
            if expected_value > best_value:
                best_value = expected_value
    
    value_cache[cache_key] = best_value
    return best_value


def terminal_value(score_diff, elo_diff=0.0):
    """
    Terminal value function for regular game end (after end 8).
    
    Parameters
    ----------
    score_diff : int
        Final score differential (ref - opp)
    elo_diff : float, default=0.0
        Elo difference (ref - opp) - not used for regular game end
    
    Returns
    -------
    float
        Win probability: 1 if ref wins, 0 if ref loses
    """
    if score_diff > 0:
        return 1.0
    elif score_diff < 0:
        return 0.0
    else:
        # Tie after end 8 goes to extra ends (handled separately)
        return 0.5  # Placeholder, will be replaced by extra_end_value


def next_hammer(current_hammer, end_diff):
    """
    Determine who has hammer next end based on mixed doubles rules.
    
    Parameters
    ----------
    current_hammer : int
        1 if ref has hammer, 0 if opp has hammer
    end_diff : int
        End differential from ref perspective (positive = ref scored)
    
    Returns
    -------
    int
        1 if ref has hammer next end, 0 if opp has hammer next end
    """
    if end_diff > 0:  # Ref scored - ref LOSES hammer, opp gets it
        return 0
    elif end_diff < 0:  # Opp scored - opp LOSES hammer, ref gets it
        return 1
    else:  # Blank end - hammer switches
        return 1 - current_hammer


def next_pp_availability(current_pp_avail, used_pp_this_end):
    """
    Update PP availability for next end.
    
    Parameters
    ----------
    current_pp_avail : int
        1 if PP available, 0 if already used
    used_pp_this_end : bool
        True if PP was used this end
    
    Returns
    -------
    int
        1 if PP available next end, 0 if not
    """
    if used_pp_this_end:
        return 0
    else:
        return current_pp_avail


def dp_value(
    end,
    score_diff,
    hammer,
    ref_pp_avail,
    opp_pp_avail,
    ep_model,
    differential_classes,
    class_to_diff,
    value_cache,
    elo_diff=0.0,
    elo_bucket_size=50.0,
    score_diff_clip=(-10, 10),
    early_quit_model=None,
    extra_end_ep_model=None,
    extra_end_differential_classes=None,
    extra_end_class_to_diff=None
):
    """
    Compute optimal value using backward induction.
    
    Note on selection bias: PP usage is not randomized. Teams choose PP based on
    game context (score, end, opponent strength, etc.). The EP model captures
    both the causal effect of PP and selection effects. We include key confounders
    (EndsRemaining, ScoreDiffStartOfEnd, EloDiff, hammer, PP availability) to mitigate this, but the
    estimated PP effect should be interpreted as observational, not causal.
    
    Parameters
    ----------
    end : int
        Current end (1-8 for regular ends, 9+ for extra ends)
    score_diff : int
        Current score differential (ref - opp), clipped to score_diff_clip range
    hammer : int
        1 if ref has hammer, 0 if opp has hammer
    ref_pp_avail : int
        1 if ref PP available, 0 if not
    opp_pp_avail : int
        1 if opp PP available, 0 if not
    ep_model : XGBClassifier
        Trained EP distribution model for regular ends
    differential_classes : np.ndarray
        Array of possible differential values for regular ends
    class_to_diff : dict
        Mapping from class index to differential value for regular ends
    value_cache : dict
        Memoization cache for computed values (key includes elo_bucket)
    elo_diff : float, default=0.0
        Elo difference (ref - opp) for EP prediction
    elo_bucket_size : float, default=50.0
        Size of Elo buckets for caching
    score_diff_clip : tuple, default=(-10, 10)
        (min, max) score differential to clip to
    early_quit_model : XGBClassifier, optional
        Model to predict early quit probability
    extra_end_ep_model : XGBClassifier, optional
        Trained EP model for extra ends
    extra_end_differential_classes : np.ndarray, optional
        Array of possible differential values for extra ends
    extra_end_class_to_diff : dict, optional
        Mapping from class index to differential value for extra ends
    
    Returns
    -------
    float
        Optimal expected win probability
    """
    # Clip score_diff to reasonable range
    score_diff = max(score_diff_clip[0], min(score_diff_clip[1], score_diff))
    
    # Terminal condition: after end 8, if tied go to extra ends
    if end > 8:
        # Extra ends: use regular EP model with EndsRemaining=0, process one end at a time
        # If someone already scored, game ends
        if score_diff != 0:
            return terminal_value(score_diff, elo_diff)
        
        # For extra ends, use regular EP model with EndsRemaining=0
        # Process one extra end at a time (continue until someone scores)
        ends_remaining = 0
        state_row = pd.Series({
            "EndsRemaining": ends_remaining,
            "RefHasHammerStartOfEnd": hammer,
            "RefScoreDiffStartOfEnd": score_diff,
            "PPUsedThisEnd": 0,
            "RefPPAvailableBeforeEnd": ref_pp_avail,
            "OppPPAvailableBeforeEnd": opp_pp_avail,
            "RefEloDiff": elo_diff
        })
        
        # Power Play cannot be used in extra ends
        actions = [0]
        
        best_value = -np.inf
        
        for use_pp in actions:
            state_row["PPUsedThisEnd"] = 0
            
            # Predict using regular EP model
            prob_dist = predict_end_differential_distribution(
                state_row, ep_model, differential_classes, class_to_diff, is_extra_end=False
            )
            
            expected_value = 0.0
            for end_diff, prob in prob_dist.items():
                if prob > 0:
                    next_score = score_diff + end_diff
                    next_hammer_val = next_hammer(hammer, end_diff)
                    
                    if hammer == 1:
                        next_ref_pp = next_pp_availability(ref_pp_avail, use_pp == 1)
                        next_opp_pp = opp_pp_avail
                    else:
                        next_ref_pp = ref_pp_avail
                        next_opp_pp = next_pp_availability(opp_pp_avail, use_pp == 1)
                    
                    # If someone scores, game ends; if blank, continue to next extra end
                    if next_score != 0:
                        # Someone scored - game ends
                        expected_value += prob * terminal_value(next_score, elo_diff)
                    else:
                        # Blank end - continue to next extra end (recursive call)
                        # Cap recursion to prevent infinite loops
                        if end < 15:  # Cap at reasonable number of extra ends
                            next_value = dp_value(
                                end + 1, next_score, next_hammer_val, next_ref_pp, next_opp_pp,
                                ep_model, differential_classes, class_to_diff, value_cache,
                                elo_diff, elo_bucket_size, score_diff_clip,
                                early_quit_model, None, None, None  # No separate extra end model
                            )
                            expected_value += prob * next_value
                        else:
                            # Too many extra ends - use Elo-based tie-breaker
                            if elo_diff == 0.0:
                                expected_value += prob * 0.5
                            else:
                                expected_value += prob * (1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0)))
            
            # For opp's turn, they minimize ref's value
            if hammer == 0 and opp_pp_avail == 1:
                if expected_value < best_value or best_value == -np.inf:
                    best_value = expected_value
            else:
                # Ref maximizes their value
                if expected_value > best_value:
                    best_value = expected_value
        
        return best_value
    
    # Check cache (include elo_diff to avoid mixing values across Elo scenarios)
    elo_bucket = round(elo_diff / elo_bucket_size) * elo_bucket_size
    state_key = (end, score_diff, hammer, ref_pp_avail, opp_pp_avail, elo_bucket)
    if state_key in value_cache:
        return value_cache[state_key]
    
    # Handle early quit probability (for ends 1-7)
    early_quit_prob = 0.0
    if early_quit_model is not None and end <= 7:
        state_row_for_quit = pd.Series({
            "EndID": end,
            "RefHasHammerStartOfEnd": hammer,
            "RefScoreDiffStartOfEnd": score_diff,
            "RefPPAvailableBeforeEnd": ref_pp_avail,
            "OppPPAvailableBeforeEnd": opp_pp_avail,
            "RefEloDiff": elo_diff
        })
        early_quit_prob = predict_early_quit_probability(state_row_for_quit, early_quit_model)
    
    # Create state row for EP prediction
    ends_remaining = max(8 - end, 0)
    state_row = pd.Series({
        "EndsRemaining": ends_remaining,
        "RefHasHammerStartOfEnd": hammer,
        "RefScoreDiffStartOfEnd": score_diff,
        "PPUsedThisEnd": 0,  # Will be set based on action: 1 if ref uses, -1 if opp uses, 0 if neither
        "RefPPAvailableBeforeEnd": ref_pp_avail,
        "OppPPAvailableBeforeEnd": opp_pp_avail,
        "RefEloDiff": elo_diff
    })
    
    # Determine available actions
    if hammer == 1 and ref_pp_avail == 1:
        # Ref has hammer and PP available: choose use_pp ∈ {0, 1}
        actions = [0, 1]  # 0 = don't use, 1 = use
    elif hammer == 0 and opp_pp_avail == 1:
        # Opp has hammer and PP available: from ref POV, opp minimizes ref's win prob
        # We model this as opp choosing use_pp ∈ {0, 1} to minimize ref's value
        actions = [0, 1]
    else:
        # No PP decision available
        actions = [0]
    
    best_value = -np.inf
    
    for use_pp in actions:
        # Set PP action in state: 1 if ref uses, -1 if opp uses, 0 if neither
        if hammer == 1:
            state_row["PPUsedThisEnd"] = 1 if use_pp == 1 else 0
        else:
            state_row["PPUsedThisEnd"] = -1 if use_pp == 1 else 0
        
        # Predict end differential distribution
        prob_dist = predict_end_differential_distribution(
            state_row, ep_model, differential_classes, class_to_diff
        )
        
        # Compute expected value over all possible outcomes
        expected_value = 0.0
        
        for end_diff, prob in prob_dist.items():
            if prob > 0:
                # Update state for next end
                next_score = score_diff + end_diff
                next_hammer_val = next_hammer(hammer, end_diff)
                
                if hammer == 1:
                    next_ref_pp = next_pp_availability(ref_pp_avail, use_pp == 1)
                    next_opp_pp = opp_pp_avail  # Opp didn't use PP
                else:
                    next_ref_pp = ref_pp_avail  # Ref didn't use PP
                    next_opp_pp = next_pp_availability(opp_pp_avail, use_pp == 1)
                
                # Check if next end is extra end (end 9)
                if end + 1 > 8:
                    # Extra end: use separate EP model
                    if extra_end_ep_model is not None:
                        next_value = extra_end_value(
                            next_score, next_hammer_val, next_ref_pp, next_opp_pp,
                            extra_end_ep_model, extra_end_differential_classes, extra_end_class_to_diff,
                            value_cache, elo_diff
                        )
                    else:
                        next_value = terminal_value(next_score, elo_diff)
                else:
                    # Regular end: recursive call
                    next_value = dp_value(
                        end + 1,
                        next_score,
                        next_hammer_val,
                        next_ref_pp,
                        next_opp_pp,
                        ep_model,
                        differential_classes,
                        class_to_diff,
                        value_cache,
                        elo_diff,
                        elo_bucket_size,
                        score_diff_clip,
                        early_quit_model,
                        extra_end_ep_model,
                        extra_end_differential_classes,
                        extra_end_class_to_diff
                    )
                
                # If game ends early after this end, use terminal value at end outcome
                if early_quit_prob > 0 and end <= 7:
                    early_quit_value = terminal_value(next_score, elo_diff)
                    expected_value += prob * early_quit_prob * early_quit_value
                
                # With probability (1 - early_quit_prob), game continues
                continue_prob = 1.0 - early_quit_prob if early_quit_prob > 0 else 1.0
                expected_value += continue_prob * prob * next_value
        
        # For opp's turn, they minimize ref's value
        if hammer == 0 and opp_pp_avail == 1:
            if expected_value < best_value or best_value == -np.inf:
                best_value = expected_value
        else:
            # Ref maximizes their value
            if expected_value > best_value:
                best_value = expected_value
    
    # Cache and return
    value_cache[state_key] = best_value
    return best_value


def pp_action_value(
    end,
    score_diff,
    hammer,
    ref_pp_avail,
    opp_pp_avail,
    use_pp,
    ep_model,
    differential_classes,
    class_to_diff,
    value_cache,
    elo_diff=0.0,
    elo_bucket_size=50.0,
    score_diff_clip=(-10, 10),
    early_quit_model=None,
    extra_end_ep_model=None,
    extra_end_differential_classes=None,
    extra_end_class_to_diff=None
):
    """
    Compute expected win probability for a forced PP action in the current end.

    This is the value of choosing use_pp (0/1) now, with optimal play afterward.
    """
    # Clip score_diff to reasonable range
    score_diff = max(score_diff_clip[0], min(score_diff_clip[1], score_diff))

    # If PP is unavailable, force use_pp to 0 to avoid invalid actions
    if hammer == 1 and ref_pp_avail == 0:
        use_pp = 0
    if hammer == 0 and opp_pp_avail == 0:
        use_pp = 0

    # Handle early quit probability (for ends 1-7)
    early_quit_prob = 0.0
    if early_quit_model is not None and end <= 7:
        state_row_for_quit = pd.Series({
            "EndID": end,
            "RefHasHammerStartOfEnd": hammer,
            "RefScoreDiffStartOfEnd": score_diff,
            "RefPPAvailableBeforeEnd": ref_pp_avail,
            "OppPPAvailableBeforeEnd": opp_pp_avail,
            "RefEloDiff": elo_diff
        })
        early_quit_prob = predict_early_quit_probability(state_row_for_quit, early_quit_model)

    # Create state row for EP prediction
    ends_remaining = max(8 - end, 0)
    state_row = pd.Series({
        "EndsRemaining": ends_remaining,
        "RefHasHammerStartOfEnd": hammer,
        "RefScoreDiffStartOfEnd": score_diff,
        "PPUsedThisEnd": 0,  # Will be set based on action
        "RefPPAvailableBeforeEnd": ref_pp_avail,
        "OppPPAvailableBeforeEnd": opp_pp_avail,
        "RefEloDiff": elo_diff
    })

    # Set PP action in state: 1 if ref uses, -1 if opp uses, 0 if neither
    if hammer == 1:
        state_row["PPUsedThisEnd"] = 1 if use_pp == 1 else 0
    else:
        state_row["PPUsedThisEnd"] = -1 if use_pp == 1 else 0

    # Predict end differential distribution
    prob_dist = predict_end_differential_distribution(
        state_row, ep_model, differential_classes, class_to_diff
    )

    expected_value = 0.0
    for end_diff, prob in prob_dist.items():
        if prob <= 0:
            continue

        next_score = score_diff + end_diff
        next_hammer_val = next_hammer(hammer, end_diff)

        if hammer == 1:
            next_ref_pp = next_pp_availability(ref_pp_avail, use_pp == 1)
            next_opp_pp = opp_pp_avail
        else:
            next_ref_pp = ref_pp_avail
            next_opp_pp = next_pp_availability(opp_pp_avail, use_pp == 1)

        next_value = dp_value(
            end + 1,
            next_score,
            next_hammer_val,
            next_ref_pp,
            next_opp_pp,
            ep_model,
            differential_classes,
            class_to_diff,
            value_cache,
            elo_diff,
            elo_bucket_size,
            score_diff_clip,
            early_quit_model,
            extra_end_ep_model,
            extra_end_differential_classes,
            extra_end_class_to_diff
        )

        if early_quit_prob > 0 and end <= 7:
            expected_value += prob * early_quit_prob * terminal_value(next_score, elo_diff)

        continue_prob = 1.0 - early_quit_prob if early_quit_prob > 0 else 1.0
        expected_value += continue_prob * prob * next_value

    return expected_value


def optimal_pp_policy(
    end,
    score_diff,
    hammer,
    ref_pp_avail,
    opp_pp_avail,
    ep_model,
    differential_classes,
    class_to_diff,
    value_cache,
    elo_diff=0.0,
    elo_bucket_size=50.0,
    score_diff_clip=(-10, 10),
    early_quit_model=None,
    extra_end_ep_model=None,
    extra_end_differential_classes=None,
    extra_end_class_to_diff=None
):
    """
    Compute optimal PP action for given state.
    
    Returns (action, value_diff) where action is 1 if should use PP, 0 if should save it,
    and value_diff is the win probability improvement from using PP.
    Only meaningful when hammer == 1 and ref_pp_avail == 1.
    
    Parameters
    ----------
    end : int
        Current end (1-8)
    score_diff : int
        Current score differential (ref - opp)
    hammer : int
        1 if ref has hammer, 0 if opp has hammer
    ref_pp_avail : int
        1 if ref PP available, 0 if not
    opp_pp_avail : int
        1 if opp PP available, 0 if not
    ep_model : XGBClassifier
        Trained EP distribution model
    differential_classes : np.ndarray
        Array of possible differential values
    class_to_diff : dict
        Mapping from class index to differential value
    value_cache : dict
        Memoization cache (shared with dp_value)
    elo_diff : float, default=0.0
        Elo difference (ref - opp) for EP prediction
    
    Returns
    -------
    tuple
        (action, value_diff) where action is 1 or 0, and value_diff is value_use - value_save
    """
    if hammer != 1 or ref_pp_avail != 1:
        return (0, 0.0)  # Not applicable

    value_use = pp_action_value(
        end,
        score_diff,
        hammer,
        ref_pp_avail,
        opp_pp_avail,
        1,
        ep_model,
        differential_classes,
        class_to_diff,
        value_cache,
        elo_diff,
        elo_bucket_size,
        score_diff_clip,
        early_quit_model,
        extra_end_ep_model,
        extra_end_differential_classes,
        extra_end_class_to_diff
    )

    value_save = pp_action_value(
        end,
        score_diff,
        hammer,
        ref_pp_avail,
        opp_pp_avail,
        0,
        ep_model,
        differential_classes,
        class_to_diff,
        value_cache,
        elo_diff,
        elo_bucket_size,
        score_diff_clip,
        early_quit_model,
        extra_end_ep_model,
        extra_end_differential_classes,
        extra_end_class_to_diff
    )

    value_diff = value_use - value_save
    action = 1 if value_use > value_save else 0
    return (action, value_diff)


def test_elo_bucket_sizes(
    end_df,
    ep_model,
    differential_classes,
    class_to_diff,
    bucket_sizes=[10.0, 25.0, 50.0, 100.0, 200.0],
    test_states=None,
    early_quit_model=None,
    extra_end_ep_model=None,
    extra_end_differential_classes=None,
    extra_end_class_to_diff=None
):
    """
    Test different Elo bucket sizes to find the most effective one.
    
    Parameters
    ----------
    end_df : pd.DataFrame
        End-level dataframe for testing
    ep_model : XGBClassifier
        Trained EP model
    differential_classes : np.ndarray
        Array of possible differential values
    class_to_diff : dict
        Mapping from class index to differential value
    bucket_sizes : list, default=[10.0, 25.0, 50.0, 100.0, 200.0]
        List of Elo bucket sizes to test
    test_states : list, optional
        List of (end, score_diff, hammer, ref_pp, opp_pp, elo_diff) tuples to test
        If None, generates default test states
    early_quit_model : XGBClassifier, optional
        Early quit model
    extra_end_ep_model : XGBClassifier, optional
        Extra end EP model
    extra_end_differential_classes : np.ndarray, optional
        Extra end differential classes
    extra_end_class_to_diff : dict, optional
        Extra end class to diff mapping
    
    Returns
    -------
    pd.DataFrame
        Results with columns: bucket_size, cache_hit_rate, avg_value_diff, num_states_tested
    """
    if test_states is None:
        # Generate default test states
        test_states = []
        for end in [1, 4, 7]:
            for score_diff in [-3, 0, 3]:
                for hammer in [0, 1]:
                    for ref_pp in [0, 1]:
                        for opp_pp in [0, 1]:
                            for elo_diff in [-100, 0, 100]:
                                test_states.append((end, score_diff, hammer, ref_pp, opp_pp, elo_diff))
    
    results = []
    score_diff_clip = (-10, 10)
    
    for bucket_size in bucket_sizes:
        value_cache = {}
        cache_hits = 0
        total_calls = 0
        value_diffs = []
        
        for end, score_diff, hammer, ref_pp, opp_pp, elo_diff in test_states:
            # Clear cache to test fresh
            value_cache.clear()
            
            # First call (should miss cache)
            v1 = dp_value(
                end, score_diff, hammer, ref_pp, opp_pp,
                ep_model, differential_classes, class_to_diff, value_cache,
                elo_diff, bucket_size, score_diff_clip,
                early_quit_model, extra_end_ep_model,
                extra_end_differential_classes, extra_end_class_to_diff
            )
            total_calls += 1
            cache_size_before = len(value_cache)
            
            # Second call (should hit cache if bucket size is good)
            value_cache2 = {}
            v2 = dp_value(
                end, score_diff, hammer, ref_pp, opp_pp,
                ep_model, differential_classes, class_to_diff, value_cache2,
                elo_diff, bucket_size, score_diff_clip,
                early_quit_model, extra_end_ep_model,
                extra_end_differential_classes, extra_end_class_to_diff
            )
            cache_size_after = len(value_cache2)
            
            # Check if values are similar (within tolerance)
            value_diff = abs(v1 - v2)
            value_diffs.append(value_diff)
            
            # Cache hit if cache size didn't grow much on second call
            if cache_size_after <= cache_size_before + 1:
                cache_hits += 1
        
        avg_value_diff = np.mean(value_diffs) if value_diffs else 0.0
        cache_hit_rate = cache_hits / len(test_states) if test_states else 0.0
        
        results.append({
            "bucket_size": bucket_size,
            "cache_hit_rate": cache_hit_rate,
            "avg_value_diff": avg_value_diff,
            "num_states_tested": len(test_states)
        })
    
    return pd.DataFrame(results)


def compute_pp_policy_heatmap(
    ep_model,
    differential_classes,
    class_to_diff,
    score_range=(-5, 5),
    elo_diff=0.0,
    opp_pp_avail=1,
    elo_bucket_size=50.0,
    score_diff_clip=(-10, 10),
    early_quit_model=None,
    extra_end_ep_model=None,
    extra_end_differential_classes=None,
    extra_end_class_to_diff=None
):
    """
    Compute optimal PP policy for all relevant states.
    
    Parameters
    ----------
    ep_model : XGBClassifier
        Trained EP distribution model
    differential_classes : np.ndarray
        Array of possible differential values
    class_to_diff : dict
        Mapping from class index to differential value
    score_range : tuple, default=(-5, 5)
        Range of score differentials to evaluate
    elo_diff : float, default=0.0
        Elo difference (ref - opp) for EP prediction
    opp_pp_avail : int, default=1
        1 if opponent PP still available (worst case), 0 if already used
    
    Returns
    -------
    pd.DataFrame
        Policy heatmap with columns: EndID, ScoreDiff, UsePP (1 or 0), ValueDiff
    """
    value_cache = {}
    policy_rows = []
    
    for end in range(1, 9):
        for score_diff in range(score_range[0], score_range[1] + 1):
            # Only compute for states where ref has hammer and PP available
            action, value_diff = optimal_pp_policy(
                end, score_diff, 1, 1, opp_pp_avail,
                ep_model, differential_classes, class_to_diff, value_cache, elo_diff,
                elo_bucket_size, score_diff_clip, early_quit_model,
                extra_end_ep_model, extra_end_differential_classes, extra_end_class_to_diff
            )
            policy_rows.append({
                "EndID": end,
                "ScoreDiff": score_diff,
                "UsePP": action,
                "ValueDiff": value_diff
            })
    
    return pd.DataFrame(policy_rows)


def plot_pp_policy_heatmap(policy_df, save_path=None):
    """
    Plot optimal PP policy heatmap showing value difference.
    Blue = Use PP (positive value), Red = Save PP (negative value).
    
    Parameters
    ----------
    policy_df : pd.DataFrame
        Policy dataframe from compute_pp_policy_heatmap()
    save_path : str, optional
        Path to save plot
    """
    # Plot value difference (win probability improvement)
    pivot_value = policy_df.pivot(index="ScoreDiff", columns="EndID", values="ValueDiff")
    plt.figure(figsize=(12, 6))
    # Blue for positive (use PP), Red for negative (save PP)
    sns.heatmap(pivot_value, cmap="RdBu", center=0, annot=True, fmt=".3f",
                cbar_kws={'label': 'ΔWP (Use PP - Save PP)'})
    plt.title("Optimal Power Play Policy\nBlue = Use PP, Red = Save PP")
    plt.xlabel("End")
    plt.ylabel("Score Differential (Ref Team)")
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_pp_delta_ep(
    end_df,
    ep_model,
    differential_classes,
    class_to_diff
):
    """
    Compute ΔEP and probability deltas (2+, 3+, steal) for PP decision.
    
    This is a simpler "good enough" approach if full DP is too slow.
    For states where ref has hammer and PP available:
    - Compute E[d | usePP] - E[d | noPP]
    - Compute ΔP(2+), ΔP(3+), ΔP(steal)
    
    Parameters
    ----------
    end_df : pd.DataFrame
        End-level dataframe (filtered to ref has hammer, PP available)
    ep_model : XGBClassifier
        Trained EP distribution model
    differential_classes : np.ndarray
        Array of possible differential values
    class_to_diff : dict
        Mapping from class index to differential value
    
    Returns
    -------
    pd.DataFrame
        Results with columns: EndID, ScoreDiff, DeltaEP, DeltaP2Plus, DeltaP3Plus, DeltaPSteal
    """
    rows = []
    
    for _, row in end_df.iterrows():
        if row["RefHasHammerStartOfEnd"] != 1 or row["RefPPAvailableBeforeEnd"] != 1:
            continue
        
        # Create state rows for use_pp and no_pp
        use_pp_row = row.copy()
        use_pp_row["PPUsedThisEnd"] = 1  # Ref uses PP
        
        no_pp_row = row.copy()
        no_pp_row["PPUsedThisEnd"] = 0  # Neither uses PP
        
        # Predict distributions
        prob_dist_use = predict_end_differential_distribution(
            use_pp_row, ep_model, differential_classes, class_to_diff
        )
        prob_dist_no = predict_end_differential_distribution(
            no_pp_row, ep_model, differential_classes, class_to_diff
        )
        
        # Compute expected differentials
        ep_use = sum(diff * prob for diff, prob in prob_dist_use.items())
        ep_no = sum(diff * prob for diff, prob in prob_dist_no.items())
        delta_ep = ep_use - ep_no
        
        # Compute probability deltas
        p2plus_use = sum(prob for diff, prob in prob_dist_use.items() if diff >= 2)
        p2plus_no = sum(prob for diff, prob in prob_dist_no.items() if diff >= 2)
        delta_p2plus = p2plus_use - p2plus_no
        
        p3plus_use = sum(prob for diff, prob in prob_dist_use.items() if diff >= 3)
        p3plus_no = sum(prob for diff, prob in prob_dist_no.items() if diff >= 3)
        delta_p3plus = p3plus_use - p3plus_no
        
        p_steal_use = sum(prob for diff, prob in prob_dist_use.items() if diff < 0)
        p_steal_no = sum(prob for diff, prob in prob_dist_no.items() if diff < 0)
        delta_p_steal = p_steal_use - p_steal_no
        
        # Compute EndsRemaining for output (if EndID exists)
        ends_remaining = max(8 - row.get("EndID", 8), 0) if "EndID" in row.index else None
        output_row = {
            "ScoreDiff": row["RefScoreDiffStartOfEnd"],
            "DeltaEP": delta_ep,
            "DeltaP2Plus": delta_p2plus,
            "DeltaP3Plus": delta_p3plus,
            "DeltaPSteal": delta_p_steal
        }
        if ends_remaining is not None:
            output_row["EndID"] = row["EndID"]
            output_row["EndsRemaining"] = ends_remaining
        rows.append(output_row)
    
    return pd.DataFrame(rows)

