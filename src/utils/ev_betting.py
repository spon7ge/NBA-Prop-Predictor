import json
from pathlib import Path

import pandas as pd

from src.utils.arbitrage import american_to_decimal, implied_prob, fmt_odds


def _load_team_games(team_source):
    """Path / str -> JSON list; DataFrame -> list of row dicts (same shape as JSON)."""
    if isinstance(team_source, pd.DataFrame):
        return team_source.to_dict("records")
    if isinstance(team_source, list):
        return team_source
    if isinstance(team_source, (str, Path)):
        with open(team_source, encoding="utf-8") as f:
            return json.load(f)
    raise TypeError(
        "team_file must be a path, list of game dicts, or DataFrame (e.g. from pd.read_json), "
        f"got {type(team_source).__name__}"
    )

"""
+EV (Positive Expected Value) Scanner
--------------------------------------
Instead of hedging both sides, you find markets where one book's
odds imply a higher win probability than the sharp consensus.

Sharp books used for "true probability" baseline: Pinnacle, LowVig, Circa
Soft books where you'll find +EV lines: DraftKings, FanDuel, BetMGM, Bovada
"""

def remove_vig(odds_a: int, odds_b: int):
    """
    Strip the vig from a 2-outcome market to get the true implied probabilities.
    Uses the standard multiplicative method.
    """
    dec_a = american_to_decimal(odds_a)
    dec_b = american_to_decimal(odds_b)
    raw_a = 1 / dec_a
    raw_b = 1 / dec_b
    total = raw_a + raw_b          # > 1.0 because of vig
    true_prob_a = raw_a / total    # normalized to sum to 1
    true_prob_b = raw_b / total
    return true_prob_a, true_prob_b


def calc_ev(your_odds: int, true_prob: float) -> float:
    """
    EV % on a $1 bet.
    true_prob = probability estimated from sharp consensus.
    """
    dec = american_to_decimal(your_odds)
    ev = (true_prob * (dec - 1)) - ((1 - true_prob) * 1)
    return ev * 100   # as a percentage


def kelly_stake(ev_pct: float, true_prob: float, your_odds: int,
                bankroll: float, kelly_fraction: float = 0.25) -> float:
    """
    Fractional Kelly — size the bet as a fraction of bankroll.
    kelly_fraction=0.25 is standard (quarter Kelly) to reduce variance.
    """
    dec = american_to_decimal(your_odds)
    b   = dec - 1          # net odds (profit per $1 risked)
    p   = true_prob
    q   = 1 - p
    kelly_pct = (b * p - q) / b
    return round(bankroll * kelly_pct * kelly_fraction, 2)


def scan_plus_ev(team_file, props_file,
                 bankroll: float = 1000,
                 min_ev: float = 2.0):
    """
    Main +EV scan.

    team_file: path to JSON, list of game dicts, DataFrame (e.g. team_dds from read_json),
        or None to skip team markets (props-only scan).
    props_file: path to CSV or DataFrame of player lines.

    For each market, uses your sharpest available books as the consensus
    to estimate true probability, then checks all soft books for +EV lines.
    """

    # Books ranked by sharpness — top of list = most trusted for true prob
    # Substring match on lowercased book name (Odds API `title` / CSV BOOKMAKER).
    SHARP_BOOKS = [
        "pinnacle",
        "circa",       
        "lowvig",      
        "betonline",   
    ]
    # Soft: retail; tokens must not falsely match SHARP (avoid a bare "bet").
    SOFT_BOOKS = [
        "draftkings",
        "fanduel",
        "betmgm",
        "betrivers",
        "bovada",
        "bally bet",
        "fliff",
        "hard rock",  
        "betparx",
        "thescore",    
        "espn bet",
        "fanatics",
        "prophetx",
        "wynnbet",
        "caesars",
        "mybookie",
        "betus",
    ]
    # ── TEAM MARKETS ──────────────────────────────────────────────────────

    ev_bets = []
    games = _load_team_games(team_file) if team_file is not None else []

    for game in games:
        matchup = f"{game['away_team']} @ {game['home_team']}"

        # Bucket sharp implied probs and soft odds per (market, point, outcome).
        # Including `point` prevents averaging across different spreads/totals (e.g. 218.5 vs 220).
        sharp_implied: dict = {}   # (mkey, point) -> outcome_name -> [implied_prob, ...]
        soft_odds:     dict = {}   # (mkey, point) -> outcome_name -> {book: american_odds}

        for bm in game["bookmakers"]:
            book = bm["bookmaker"].lower()
            for market in bm["markets"]:
                mkey = market["market_key"]
                for outcome in market["outcomes"]:
                    name  = outcome["name"]
                    price = outcome["price"]
                    point = outcome.get("point")
                    key   = (mkey, point)

                    if any(s in book for s in SHARP_BOOKS):
                        sharp_implied.setdefault(key, {}).setdefault(name, []).append(implied_prob(price))

                    if any(s in book for s in SOFT_BOOKS):
                        soft_odds.setdefault(key, {}).setdefault(name, {})[book] = price

        # Average sharp PROBABILITIES (not American odds), then normalize the pair to vig-free true probs.
        for (mkey, point), outcomes in sharp_implied.items():
            names = list(outcomes.keys())
            if len(names) != 2:
                continue

            a, b = names
            avg_a = sum(outcomes[a]) / len(outcomes[a])
            avg_b = sum(outcomes[b]) / len(outcomes[b])
            total = avg_a + avg_b
            if total <= 0:
                continue
            true_prob_a = avg_a / total
            true_prob_b = avg_b / total

            market_label = mkey if point is None else f"{mkey} {point:+}"

            for side, true_prob in [(a, true_prob_a), (b, true_prob_b)]:
                for book, book_odds in soft_odds.get((mkey, point), {}).get(side, {}).items():
                    ev = calc_ev(book_odds, true_prob)
                    if ev >= min_ev:
                        stake = kelly_stake(ev, true_prob, book_odds, bankroll)
                        ev_bets.append({
                            "type": "team", "game": matchup, "market": market_label,
                            "bet": side, "book": book,
                            "odds": fmt_odds(book_odds),
                            "true_prob": round(true_prob * 100, 1),
                            "implied_prob": round(implied_prob(book_odds) * 100, 1),
                            "ev_pct": round(ev, 2),
                            "kelly_stake": stake,
                        })

    # ── PLAYER PROPS ──────────────────────────────────────────────────────

    df = props_file.copy() if isinstance(props_file, pd.DataFrame) else pd.read_csv(props_file)
    df["decimal"] = df["ODDS"].apply(american_to_decimal)
    df["book_lower"] = df["BOOKMAKER"].str.lower()

    sharp_props = df[df["book_lower"].apply(lambda b: any(s in b for s in SHARP_BOOKS))].copy()
    soft_props  = df[df["book_lower"].apply(lambda b: any(s in b for s in SOFT_BOOKS))]

    # Average implied prob (1/decimal) per sharp line — avoids bias from averaging American odds
    sharp_props["implied"] = 1.0 / sharp_props["decimal"]
    sharp_avg = (
        sharp_props.groupby(["NAME", "CATEGORY", "LINE", "OVER/UNDER"], as_index=False)["implied"]
        .mean()
    )

    # Pair Over/Under; normalize averaged implieds to vig-free true probs (sum to 1)
    sharp_over = sharp_avg[sharp_avg["OVER/UNDER"] == "Over"].rename(columns={"implied": "implied_over"})
    sharp_under = sharp_avg[sharp_avg["OVER/UNDER"] == "Under"].rename(columns={"implied": "implied_under"})
    sharp_paired = sharp_over.merge(sharp_under, on=["NAME", "CATEGORY", "LINE"])

    if not sharp_paired.empty:
        tot = sharp_paired["implied_over"] + sharp_paired["implied_under"]
        sharp_paired = sharp_paired.loc[tot > 0].copy()
        tot = sharp_paired["implied_over"] + sharp_paired["implied_under"]
        sharp_paired["true_prob_over"] = sharp_paired["implied_over"] / tot
        sharp_paired["true_prob_under"] = sharp_paired["implied_under"] / tot

        # Check each soft book line against true prob
        for _, sp in sharp_paired.iterrows():
            for side, true_prob in [("Over", sp["true_prob_over"]), ("Under", sp["true_prob_under"])]:
                soft_lines = soft_props[
                    (soft_props["NAME"]       == sp["NAME"]) &
                    (soft_props["CATEGORY"]   == sp["CATEGORY"]) &
                    (soft_props["LINE"]       == sp["LINE"]) &
                    (soft_props["OVER/UNDER"] == side)
                ]
                for _, row in soft_lines.iterrows():
                    ev = calc_ev(int(row["ODDS"]), true_prob)
                    if ev >= min_ev:
                        stake = kelly_stake(ev, true_prob, int(row["ODDS"]), bankroll)
                        ev_bets.append({
                            "type":        "prop",
                            "player":      sp["NAME"],
                            "category":    sp["CATEGORY"],
                            "line":        sp["LINE"],
                            "bet":         f"{side} {sp['LINE']}",
                            "book":        row["BOOKMAKER"],
                            "odds":        fmt_odds(int(row["ODDS"])),
                            "true_prob":   round(true_prob * 100, 1),
                            "implied_prob": round(implied_prob(int(row["ODDS"])) * 100, 1),
                            "ev_pct":      round(ev, 2),
                            "kelly_stake": stake,
                        })

    return sorted(ev_bets, key=lambda r: r["ev_pct"], reverse=True)


# ── display ───────────────────────────────────────────────────────────────────

def print_ev_results(ev_bets: list, bankroll: float):
    print(f"\n{'='*60}")
    print(f"  +EV SCANNER  |  bankroll=${bankroll:.0f}")
    print(f"{'='*60}")
    print(f"  +EV bets found: {len(ev_bets)}\n")

    for r in ev_bets:
        if r["type"] == "prop":
            title = f"[PROP] {r['player']} — {r['category']} {r['bet']}"
        else:
            title = f"[TEAM] {r['game']} — {r['market']} — {r['bet']}"

        print(title)
        print(f"  Book: {r['book']}  |  Odds: {r['odds']}")
        print(f"  True prob: {r['true_prob']}%  vs  Implied: {r['implied_prob']}%")
        print(f"  EV: +{r['ev_pct']}%  |  Kelly stake: ${r['kelly_stake']}")
        print()