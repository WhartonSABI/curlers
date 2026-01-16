"""
Streamlit app for interactive PP recommendations.

Run locally:
  streamlit run code/pp_recommender_app.py
"""

import os

import pandas as pd
import streamlit as st

from data_prep import load_data, prepare_ends, build_start_of_end_df
from ep_end import (
    train_end_differential_distribution_model,
    train_early_quit_model,
    predict_end_differential_distribution,
)
from ep_policy import dp_value, next_hammer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_END_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "end_level_start.csv")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

SCORE_DIFF_CLIP = (-10, 10)
DEFAULT_ELO_BUCKET = 50.0


@st.cache_data(show_spinner=False)
def load_end_level_df():
    """Load precomputed end-level data if available, else build from raw."""
    if os.path.exists(PROCESSED_END_PATH):
        return pd.read_csv(PROCESSED_END_PATH, low_memory=False)

    stones, _, games, ends, _, _ = load_data(data_dir=RAW_DATA_DIR)
    ends_prep = prepare_ends(ends)
    return build_start_of_end_df(ends_prep, stones, games)


@st.cache_resource(show_spinner=True)
def load_models():
    """Train EP and early quit models once per session."""
    end_df = load_end_level_df()
    ep_model, _, _, differential_classes, class_to_diff = train_end_differential_distribution_model(
        end_df, is_extra_end=False
    )
    try:
        early_quit_model, _, _ = train_early_quit_model(end_df)
    except Exception:
        early_quit_model = None
    return ep_model, differential_classes, class_to_diff, early_quit_model


def compute_use_save_values(
    end_id,
    score_diff,
    opp_pp_avail,
    elo_diff,
    ep_model,
    differential_classes,
    class_to_diff,
    early_quit_model,
    value_cache,
):
    """Compute WP if use PP now vs save PP for later."""
    ends_remaining = max(8 - end_id, 0)
    base_row = pd.Series(
        {
            "EndsRemaining": ends_remaining,
            "RefHasHammerStartOfEnd": 1,
            "RefScoreDiffStartOfEnd": score_diff,
            "PPUsedThisEnd": 0,
            "RefPPAvailableBeforeEnd": 1,
            "OppPPAvailableBeforeEnd": opp_pp_avail,
            "RefEloDiff": elo_diff,
        }
    )

    # Use PP now
    base_row["PPUsedThisEnd"] = 1
    prob_dist_use = predict_end_differential_distribution(
        base_row, ep_model, differential_classes, class_to_diff
    )
    value_use = 0.0
    for end_diff, prob in prob_dist_use.items():
        if prob <= 0:
            continue
        next_score = score_diff + end_diff
        next_hammer_val = next_hammer(1, end_diff)
        next_ref_pp = 0
        next_opp_pp = opp_pp_avail
        value_use += prob * dp_value(
            end_id + 1,
            next_score,
            next_hammer_val,
            next_ref_pp,
            next_opp_pp,
            ep_model,
            differential_classes,
            class_to_diff,
            value_cache,
            elo_diff,
            DEFAULT_ELO_BUCKET,
            SCORE_DIFF_CLIP,
            early_quit_model,
            None,
            None,
            None,
        )

    # Save PP for later
    base_row["PPUsedThisEnd"] = 0
    prob_dist_save = predict_end_differential_distribution(
        base_row, ep_model, differential_classes, class_to_diff
    )
    value_save = 0.0
    for end_diff, prob in prob_dist_save.items():
        if prob <= 0:
            continue
        next_score = score_diff + end_diff
        next_hammer_val = next_hammer(1, end_diff)
        next_ref_pp = 1
        next_opp_pp = opp_pp_avail
        value_save += prob * dp_value(
            end_id + 1,
            next_score,
            next_hammer_val,
            next_ref_pp,
            next_opp_pp,
            ep_model,
            differential_classes,
            class_to_diff,
            value_cache,
            elo_diff,
            DEFAULT_ELO_BUCKET,
            SCORE_DIFF_CLIP,
            early_quit_model,
            None,
            None,
            None,
        )

    return value_use, value_save


def main():
    st.set_page_config(page_title="Power Play Recommender", layout="centered")
    st.title("Power Play Recommender")
    st.write(
        "Enter the current game state and team strength information to get a "
        "WP-optimized PP recommendation."
    )

    with st.sidebar:
        st.header("Game state")
        end_id = st.slider("End", min_value=1, max_value=8, value=4, step=1)
        score_diff = st.number_input(
            "Score Differential (Ref - Opp)",
            min_value=SCORE_DIFF_CLIP[0],
            max_value=SCORE_DIFF_CLIP[1],
            value=0,
            step=1,
        )
        ref_has_hammer = st.radio("Does Ref team have hammer?", ["Yes", "No"], index=0)
        ref_pp_avail = st.radio("Does Ref team have PP available?", ["Yes", "No"], index=0)
        opp_pp_avail = st.radio("Does Opp have PP available?", ["Yes", "No"], index=0)
        st.header("Team Strength")
        elo_diff = st.number_input(
            "Elo Differential (Ref - Opp)",
            min_value=-400,
            max_value=400,
            value=0,
            step=10,
        )

    hammer_flag = 1 if ref_has_hammer == "Yes" else 0
    ref_pp_flag = 1 if ref_pp_avail == "Yes" else 0
    opp_pp_flag = 1 if opp_pp_avail == "Yes" else 0

    with st.spinner("Loading models..."):
        ep_model, differential_classes, class_to_diff, early_quit_model = load_models()

    if "value_cache" not in st.session_state:
        st.session_state["value_cache"] = {}
    value_cache = st.session_state["value_cache"]

    if hammer_flag != 1 or ref_pp_flag != 1:
        st.info(
            "No PP decision is available for this state. "
            "PP decisions only apply when the Ref team has hammer and "
            "their PP is still available."
        )
        return

    value_use, value_save = compute_use_save_values(
        end_id,
        int(score_diff),
        opp_pp_flag,
        float(elo_diff),
        ep_model,
        differential_classes,
        class_to_diff,
        early_quit_model,
        value_cache,
    )
    value_diff = value_use - value_save
    recommended = "Use PP now" if value_diff > 0 else "Save PP for later"
    if value_diff > 0:
        diff_color = "#2e7d32"
    elif value_diff < 0:
        diff_color = "#c62828"
    else:
        diff_color = "#616161"

    col_use, col_save, col_delta = st.columns(3)

    label_style = "font-size: 1.6rem; font-weight: 600; margin: 0 0 0.2rem 0;"
    value_style = "font-size: 2.2rem; font-weight: 700; margin: 0;"

    col_use.markdown(
        f"<div style='{label_style}'>WP if use PP</div>",
        unsafe_allow_html=True,
    )
    col_use.markdown(
        f"<div style='{value_style}'>{value_use:.2%}</div>",
        unsafe_allow_html=True,
    )

    col_save.markdown(
        f"<div style='{label_style}'>WP if save PP</div>",
        unsafe_allow_html=True,
    )
    col_save.markdown(
        f"<div style='{value_style}'>{value_save:.2%}</div>",
        unsafe_allow_html=True,
    )

    col_delta.markdown(
        f"<div style='{label_style}'>WP Δ (use - save)</div>",
        unsafe_allow_html=True,
    )
    col_delta.markdown(
        "<div style='{style} color: {color};'>{value}</div>".format(
            style=value_style,
            color=diff_color,
            value=f"{value_diff:+.2%}",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Recommendation: "
        f"<span style='color: {diff_color}; font-weight: 700;'>{recommended}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "WP values are model-based estimates. Positive Δ favors using PP now; "
        "negative Δ favors saving it."
    )


if __name__ == "__main__":
    main()
