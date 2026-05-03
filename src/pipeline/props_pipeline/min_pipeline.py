import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.team_info import projectedStartingFive
from src.utils.helpers import load_team_odds

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


def min_pipeline(df, name, current_date):
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]
    player_team = last["TEAM_ABBREVIATION"]
    team_name = last["TEAM_NAME"]

    # STARTING_X_SPREAD_ABS — matches training: STARTING * |TEAM_SPREAD|
    projected = projectedStartingFive.get(player_team, [])
    starting_override = float(1 if (name in projected) else 0)
    odds_df = _cached_team_odds()
    spread_pt = _player_team_spread(odds_df, str(team_name), str(player_team))
    if spread_pt is not None and pd.notna(spread_pt):
        res.append(float(starting_override * abs(spread_pt)))
    else:
        res.append(float("nan"))

    # MIN_10_ewm
    min_series = pdf["MIN"].astype(float)
    min_10_ewm = float(min_series.ewm(span=10, adjust=False).mean().iloc[-1])
    res.append(min_10_ewm if pd.notna(min_10_ewm) else float("nan"))

    # TEAM_MIN_RANK_L10
    team_df = df[df["TEAM_ID"] == last["TEAM_ID"]]
    teammate_roll = {}
    for player, grp in team_df.groupby("PLAYER_NAME"):
        grp_sorted = grp.sort_values("GAME_DATE")
        roll_avg = grp_sorted["MIN"].astype(float).tail(10).mean()
        teammate_roll[player] = roll_avg
    sorted_teammates = sorted(teammate_roll.items(), key=lambda x: x[1], reverse=True)
    rank_map = {p: float(i + 1) for i, (p, _) in enumerate(sorted_teammates)}
    team_min_rank_l10 = rank_map.get(name, float("nan"))
    res.append(team_min_rank_l10 if pd.notna(team_min_rank_l10) else float("nan"))

    # STARTER_ROLL10_PCT
    starter_roll10_pct = float(pdf["STARTING"].tail(10).mean())
    res.append(starter_roll10_pct if pd.notna(starter_roll10_pct) else float("nan"))

    # MIN_SEASON_MEAN
    season_mean = float(pdf["MIN"].mean())
    res.append(season_mean if pd.notna(season_mean) else float("nan"))

    # MIN_lag1
    res.append(float(pdf["MIN"].iloc[-1]))

    # MIN_SEASON_STD
    season_std = float(pdf["MIN"].std())
    res.append(season_std if pd.notna(season_std) else float("nan"))

    # MIN_MAX_L10
    res.append(float(pdf["MIN"].tail(10).max()))

    # IS_PLAYOFF
    is_playoff = float(last["IS_PLAYOFF"]) if "IS_PLAYOFF" in last.index else float("nan")
    res.append(is_playoff)

    return res