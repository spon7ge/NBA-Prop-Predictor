import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.helper_functions import findOpp
from src.utils.team_info import projectedStartingFive
from src.utils.helpers import load_team_odds

from src.pipeline.props_pipeline.ppm_pipeline import (
    _active_stars_from_projected_lineup,
    _opp_roll_metric,
    _team_game_logs,
)

# Order must match training / saved quantile bundle `feature_names`.
MIN_FEATURES = [
    "STARTING_X_SPREAD_ABS",
    "MIN_5_ewm",
    "TEAM_MIN_RANK_L10",
    "STARTER_ROLL10_PCT",
    "MIN_SEASON_MEAN",
    "MIN_lag1",
    "MIN_SEASON_STD",
    "MIN_MAX_L10",
    "IS_PLAYOFF",
    "MIN_deviation",
    "ACTIVE_STARS_COUNT",
    "EXPECTED_PACE",
    "MIN_TREND",
]

# Reload team odds when a newer JSON lands (same process predicts many players).
_odds_mtime: float | None = None
_odds_df: pd.DataFrame | None = None


def _cached_team_odds() -> pd.DataFrame | None:
    global _odds_mtime, _odds_df
    files = list(Path("data/raw/team_lines").glob("NBA_*.json"))
    if not files:
        return None
    newest = max(files, key=lambda f: f.stat().st_mtime)
    mt = newest.stat().st_mtime
    if _odds_df is None or _odds_mtime != mt:
        _odds_mtime = mt
        _odds_df = load_team_odds()
    return _odds_df


def _player_team_spread(
    team_odds: pd.DataFrame | None,
    team_name: str,
    team_abbrev: str,
) -> float | None:
    """
    Spread *point* for the player's team from their game in team odds.
    Book priority: DraftKings, then BetMGM.
    """
    if team_odds is None or team_odds.empty:
        return None

    games = team_odds.to_dict("records")

    def _norm(s: str) -> str:
        return str(s).lower().replace(".", "").replace("-", " ").strip()

    def _team_matches(feed_name: str) -> bool:
        n = _norm(feed_name)
        if n == _norm(team_name):
            return True
        if team_abbrev and n == _norm(team_abbrev):
            return True
        return False

    game_data = next(
        (
            g
            for g in games
            if _team_matches(g.get("home_team", ""))
            or _team_matches(g.get("away_team", ""))
        ),
        None,
    )
    if game_data is None:
        return None

    books = {b.get("bookmaker"): b for b in game_data.get("bookmakers", [])}
    bk = None
    for book in ("DraftKings", "BetMGM"):
        if book in books:
            bk = books[book]
            break
    if bk is None:
        return None

    for market in bk.get("markets", []):
        if market.get("market_key") != "spreads":
            continue
        for outcome in market.get("outcomes", []):
            if _team_matches(outcome.get("name", "")):
                pt = outcome.get("point")
                if pt is None:
                    return None
                return float(pt)
    return None


def _next_row_shift_ewm(s: pd.Series, span: int) -> float:
    """shift(1).ewm (min_features) evaluated for the row after the last observation."""
    ext = pd.concat([s.astype(float).reset_index(drop=True), pd.Series([np.nan])], ignore_index=True)
    v = ext.shift(1).ewm(span=span, adjust=False).mean().round(2).iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _next_row_shift_roll_mean(
    s: pd.Series, window: int, min_periods: int = 3, round_ndigits: int | None = None
) -> float:
    ext = pd.concat([s.astype(float).reset_index(drop=True), pd.Series([np.nan])], ignore_index=True)
    raw = ext.shift(1).rolling(window, min_periods=min_periods).mean()
    v = raw.iloc[-1]
    if round_ndigits is not None and pd.notna(v):
        v = round(float(v), round_ndigits)
    return float(v) if pd.notna(v) else float("nan")


def _next_row_shift_roll_max(s: pd.Series, window: int) -> float:
    """Matches notebook: shift(1).rolling(10).max() for the upcoming game row."""
    ext = pd.concat([s.astype(float).reset_index(drop=True), pd.Series([np.nan])], ignore_index=True)
    v = ext.shift(1).rolling(window).max().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def min_pipeline(df, name, current_date):
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]
    player_team = last["TEAM_ABBREVIATION"]
    team_name = last["TEAM_NAME"]
    min_s = pdf["MIN"].astype(float)

    # STARTING_X_SPREAD_ABS — matches training: STARTING * |TEAM_SPREAD|
    projected = projectedStartingFive.get(player_team, [])
    starting_override = float(1 if (name in projected) else 0)
    odds_df = _cached_team_odds()
    spread_pt = _player_team_spread(odds_df, str(team_name), str(player_team))
    if spread_pt is not None and pd.notna(spread_pt):
        res.append(float(starting_override * abs(spread_pt)))
    else:
        res.append(float("nan"))

    # MIN_5_ewm (min_features._ewm_player)
    min_5_ewm_val = _next_row_shift_ewm(min_s, 5)
    res.append(min_5_ewm_val)

    # TEAM_MIN_RANK_L10 — rank by MIN_roll10 (dense, high MIN = rank 1)
    team_df = df[df["TEAM_ID"] == last["TEAM_ID"]]
    teammate_roll = {}
    for player, grp in team_df.groupby("PLAYER_NAME"):
        g_sorted = grp.sort_values("GAME_DATE")
        teammate_roll[player] = _next_row_shift_roll_mean(g_sorted["MIN"].astype(float), 10, min_periods=3)
    sorted_teammates = sorted(
        teammate_roll.items(),
        key=lambda it: it[1] if pd.notna(it[1]) else -np.inf,
        reverse=True,
    )
    rank_map = {p: float(i + 1) for i, (p, _) in enumerate(sorted_teammates)}
    team_min_rank = rank_map.get(name, float("nan"))
    res.append(team_min_rank if pd.notna(team_min_rank) else float("nan"))

    # STARTER_ROLL10_PCT (min_features._starter_features)
    if "STARTING" not in pdf.columns:
        res.append(float("nan"))
    else:
        st = pdf["STARTING"].astype(float)
        res.append(_next_row_shift_roll_mean(st, 10, min_periods=3, round_ndigits=2))

    # MIN_SEASON_MEAN — notebook: groupby PLAYER_ID MIN expanding mean shift(1); next row ⇒ mean(all)
    season_mean = float(min_s.mean())
    res.append(season_mean if pd.notna(season_mean) else float("nan"))

    # MIN_lag1 — next row ⇒ last observed MIN
    res.append(float(min_s.iloc[-1]))

    # MIN_SEASON_STD — notebook: expanding std shift(1); next row ⇒ std(all, ddof=1)
    if len(min_s) > 1:
        season_std = float(min_s.std(ddof=1))
    else:
        season_std = float("nan")
    res.append(season_std)

    # MIN_MAX_L10
    res.append(_next_row_shift_roll_max(min_s, 10))

    # IS_PLAYOFF
    is_playoff = float(last["IS_PLAYOFF"]) if "IS_PLAYOFF" in last.index else float("nan")
    res.append(is_playoff)

    # MIN_deviation — notebook: MIN_lag1 - MIN_roll10 (next-row aligned)
    min_lag1 = float(min_s.iloc[-1])
    min_roll10 = _next_row_shift_roll_mean(min_s, 10, min_periods=3)
    res.append(min_lag1 - min_roll10 if pd.notna(min_roll10) else float("nan"))

    # ACTIVE_STARS_COUNT — team3Stars present in projectedStartingFive (see team_info)
    res.append(float(_active_stars_from_projected_lineup(str(player_team))))

    # EXPECTED_PACE — min_features._expectedPace
    opp_abbr, _ = findOpp(name, df, current_date, max_days_ahead=3)
    team_pace_r10 = _opp_roll_metric(_team_game_logs(df, str(player_team)), "TEAM_PACE", 10)
    opp_pace_r10 = float("nan")
    if opp_abbr is not None:
        opp_pace_r10 = _opp_roll_metric(_team_game_logs(df, opp_abbr), "TEAM_PACE", 10)
    if pd.notna(team_pace_r10) and pd.notna(opp_pace_r10):
        res.append(round((team_pace_r10 + opp_pace_r10) / 2, 2))
    else:
        res.append(float("nan"))

    # MIN_TREND — notebook: MIN_5_ewm - MIN_SEASON_MEAN
    if pd.notna(min_5_ewm_val) and pd.notna(season_mean):
        res.append(float(min_5_ewm_val - season_mean))
    else:
        res.append(float("nan"))

    return res
