from __future__ import annotations

_PP_ALIASES: dict[str, str] = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "3-pt_made": "threes",
    "3_pt_made": "threes",
    "pts_rebs": "pts_rebs",
    "pts_asts": "pts_asts",
    "rebs_asts": "rebs_asts",
    "pts_rebs_asts": "pts_rebs_asts",
}

_UD_ALIASES: dict[str, str] = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "three_points_made": "threes",
    "pts_rebs": "pts_rebs",
    "pts_asts": "pts_asts",
    "rebs_asts": "rebs_asts",
    "pts_rebs_asts": "pts_rebs_asts",
}

_PARLAY_ALIASES: dict[str, str] = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_three_pointers": "threes",
    "player_three_pointers_made": "threes",
    "player_pts_rebs": "pts_rebs",
    "player_points_rebounds": "pts_rebs",
    "player_pts_asts": "pts_asts",
    "player_points_assists": "pts_asts",
    "player_rebs_asts": "rebs_asts",
    "player_assists_rebounds": "rebs_asts",
    "player_pra": "pts_rebs_asts",
    "player_pts_rebs_asts": "pts_rebs_asts",
    "player_points_rebounds_assists": "pts_rebs_asts",
}

_LABELS: dict[str, str] = {
    "points": "Points",
    "rebounds": "Rebounds",
    "assists": "Assists",
    "threes": "3-PT Made",
    "pts_rebs": "Pts+Rebs",
    "pts_asts": "Pts+Asts",
    "rebs_asts": "Rebs+Asts",
    "pts_rebs_asts": "Pts+Rebs+Asts",
}


def _norm_pp(stat_type: str) -> str:
    return stat_type.strip().lower().replace(" ", "_").replace("+", "_")


def canonical_stat_key_from_pp(stat_type: str) -> str | None:
    return _PP_ALIASES.get(_norm_pp(stat_type))


def canonical_stat_key_from_ud(stat_name: str) -> str | None:
    return _UD_ALIASES.get(stat_name.strip().lower().replace(" ", "_"))


def canonical_stat_key_from_parlay_market(market_key: str) -> str | None:
    return _PARLAY_ALIASES.get(market_key.strip().lower())


def display_stat_label(stat_key: str, fallback: str | None = None) -> str:
    return _LABELS.get(stat_key) or fallback or stat_key.replace("_", " ").title()
