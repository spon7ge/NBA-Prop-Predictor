from app.services.prop_stat_keys import (
    canonical_stat_key_from_pp,
    canonical_stat_key_from_ud,
    canonical_stat_key_from_parlay_market,
)


def test_pp_core_stats():
    assert canonical_stat_key_from_pp("Points") == "points"
    assert canonical_stat_key_from_pp("Pts+Rebs+Asts") == "pts_rebs_asts"
    assert canonical_stat_key_from_pp("3-PT Made") == "threes"
    assert canonical_stat_key_from_pp("Pts+Rebs") == "pts_rebs"


def test_pp_unmatched_returns_none():
    assert canonical_stat_key_from_pp("Fantasy Score") is None
    assert canonical_stat_key_from_pp("Points (Combo)") is None
    assert canonical_stat_key_from_pp("Defensive Rebounds") is None


def test_ud_core_stats():
    assert canonical_stat_key_from_ud("points") == "points"
    assert canonical_stat_key_from_ud("three_points_made") == "threes"
    assert canonical_stat_key_from_ud("pts_rebs_asts") == "pts_rebs_asts"


def test_parlay_markets():
    assert canonical_stat_key_from_parlay_market("player_points") == "points"
    assert canonical_stat_key_from_parlay_market("player_threes") == "threes"
    assert canonical_stat_key_from_parlay_market("player_three_pointers_made") == "threes"
    assert canonical_stat_key_from_parlay_market("player_pra") == "pts_rebs_asts"
    assert canonical_stat_key_from_parlay_market("player_points_rebounds_assists") == "pts_rebs_asts"
