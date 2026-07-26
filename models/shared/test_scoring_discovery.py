"""Tests for models.shared.scoring_discovery helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.shared import scoring_discovery as sd


def test_derive_pts_multiplies_rate_by_minutes():
    df = pd.DataFrame({"pts_per_min": [0.5, 1.0], "minutes": [20.0, 30.0]})
    out = sd.derive_pts(df)
    assert out["pts"].tolist() == pytest.approx([10.0, 30.0])


def test_build_coverage_map_statuses():
    cols = [
        "fga_per_min",
        "base_fga_per_min_ewm_hl10",
        "adv_usg_pct_season_avg",
        "track_tchs_per_min_ewm_hl10",
        "opp_def_rating_ewm_hl10",
        "days_rest",
        "is_back_to_back",
    ]
    cov = sd.build_coverage_map(cols)
    assert set(cov["status"]) <= {"available", "partial", "missing"}
    usg = cov.loc[cov["concept"].str.contains("Usage", case=False)].iloc[0]
    assert usg["status"] in {"available", "partial"}
    matchup = cov.loc[cov["concept"].str.contains("Primary defender", case=False)]
    assert not matchup.empty
    assert matchup.iloc[0]["status"] == "missing"


def test_split_feature_pools_keeps_prior_out_of_same_game():
    cols = [
        "fga_per_min",
        "tchs_per_min",
        "base_fga_per_min_ewm_hl10",
        "adv_usg_pct_season_avg",
        "opp_pace_ewm_hl10",
        "days_rest",
        "player_id",
        "pts_per_min",
        "pts",
        "minutes",
    ]
    pools = sd.split_feature_pools(cols, targets=["pts", "pts_per_min", "minutes"])
    assert "fga_per_min" in pools["same_game"]
    assert "base_fga_per_min_ewm_hl10" in pools["predictive"]
    assert "fga_per_min" not in pools["predictive"]
    assert "pts" in pools["excluded"]
    assert "player_id" in pools["excluded"]


def test_lineage_for_prefixes():
    assert sd.lineage_for("base_pts_per_min_ewm_hl5") == "prior_player"
    assert sd.lineage_for("opp_def_rating_ewm_hl10") == "opponent"
    assert sd.lineage_for("team_pace_ewm_hl10") == "team"
    assert sd.lineage_for("days_rest") == "context"
    assert sd.lineage_for("fga_per_min") == "same_game"


def test_rank_univariate_orders_strong_signal_first():
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    df = pd.DataFrame({
        "signal": signal,
        "noise": noise,
        "y": signal * 2 + rng.normal(scale=0.1, size=n),
    })
    ranks = sd.rank_univariate(df, ["signal", "noise"], "y", random_state=42)
    assert ranks.iloc[0]["feature"] == "signal"
    assert abs(ranks.loc[ranks["feature"] == "signal", "spearman"].iloc[0]) > 0.8


def test_season_rank_stability_returns_score():
    rng = np.random.default_rng(1)
    rows = []
    for season in ("2023-24", "2024-25"):
        for _ in range(200):
            s = rng.normal()
            rows.append({
                "season_year": season,
                "good": s,
                "noise": rng.normal(),
                "y": s + rng.normal(scale=0.2),
            })
    df = pd.DataFrame(rows)
    stab = sd.season_rank_stability(
        df, ["good", "noise"], "y", season_col="season_year", top_n=2, random_state=42,
    )
    assert "stability" in stab.columns
    assert stab.loc[stab["feature"] == "good", "stability"].iloc[0] >= \
        stab.loc[stab["feature"] == "noise", "stability"].iloc[0]


def test_merge_driver_shortlist_intersects_signals():
    uni = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "spearman": [0.9, 0.2, 0.5],
        "mi": [0.8, 0.1, 0.4],
        "lineage": ["prior_player"] * 3,
    })
    shap = pd.DataFrame({"feature": ["a", "c", "b"], "mean_abs_shap": [1.0, 0.5, 0.1]})
    out = sd.merge_driver_shortlist(uni, shap_df=shap, top_k=2)
    assert list(out["feature"])[:2] == ["a", "c"]
