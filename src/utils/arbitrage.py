import pandas as pd

def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1
 
 
def implied_prob(odds: int) -> float:
    return 1 / american_to_decimal(odds)
 
 
def check_arb(odds_a: int, odds_b: int):
    """
    Given best odds on each side of a 2-outcome market,
    return (margin_pct, stake_a, stake_b) for a $100 total stake.
    margin_pct > 0 means a real arb exists.
    """
    dec_a = american_to_decimal(odds_a)
    dec_b = american_to_decimal(odds_b)
    total_implied = (1 / dec_a) + (1 / dec_b)
    margin_pct = (1 - total_implied) * 100
 
    # Kelly-optimal split for guaranteed return
    weight_a = (1 / dec_a) / total_implied
    weight_b = 1 - weight_a
    return margin_pct, weight_a, weight_b
 
 
def fmt_odds(o: int) -> str:
    return f"+{o}" if o > 0 else str(o)
 
 
def calc_bets(total_stake: float, weight_a: float, weight_b: float,
              dec_a: float, dec_b: float):
    stake_a = total_stake * weight_a
    stake_b = total_stake * weight_b
    profit = (stake_a * dec_a) - total_stake   # same whichever side wins
    return round(stake_a, 2), round(stake_b, 2), round(profit, 2)

MIN_MARGIN = 0.0
TOTAL_STAKE = 100.0


def scan_player_props(csv_path):
    df = pd.read_csv(csv_path)
    arbs = []
    grouped = df.groupby(["NAME", "CATEGORY", "LINE"])
    for (name, category, line), g in grouped:
        overs = g[g["OVER/UNDER"].str.lower() == "over"]
        unders = g[g["OVER/UNDER"].str.lower() == "under"]
        if overs.empty or unders.empty:
            continue
        best_over = overs.loc[overs["ODDS"].idxmax()]
        best_under = unders.loc[unders["ODDS"].idxmax()]
        margin, w_o, w_u = check_arb(int(best_over["ODDS"]), int(best_under["ODDS"]))
        if margin > MIN_MARGIN:
            dec_o = american_to_decimal(int(best_over["ODDS"]))
            dec_u = american_to_decimal(int(best_under["ODDS"]))
            stake_o, stake_u, profit = calc_bets(TOTAL_STAKE, w_o, w_u, dec_o, dec_u)
            arbs.append({
                "type": "prop",
                "market": f"{name} {category} {line}",
                "side_a": f"Over @ {best_over['BOOKMAKER']} ({fmt_odds(int(best_over['ODDS']))})",
                "side_b": f"Under @ {best_under['BOOKMAKER']} ({fmt_odds(int(best_under['ODDS']))})",
                "margin_%": round(margin, 3),
                "stake_a": stake_o,
                "stake_b": stake_u,
                "profit": profit,
            })
    return arbs


def scan_team_odds(team_df):
    arbs = []
    for _, row in team_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        # collect best price per (market, side, point) across books
        best = {}  # key: (market_key, name, point) -> (price, book)
        for bk in row["bookmakers"]:
            book = bk["bookmaker"]
            for mkt in bk["markets"]:
                mkey = mkt["market_key"]
                for o in mkt["outcomes"]:
                    key = (mkey, o["name"], o.get("point"))
                    if key not in best or o["price"] > best[key][0]:
                        best[key] = (o["price"], book)

        # totals: pair Over/Under at same point
        totals_points = {p for (m, n, p) in best if m == "totals"}
        for pt in totals_points:
            o = best.get(("totals", "Over", pt))
            u = best.get(("totals", "Under", pt))
            if not o or not u:
                continue
            margin, w_o, w_u = check_arb(o[0], u[0])
            if margin > MIN_MARGIN:
                dec_o, dec_u = american_to_decimal(o[0]), american_to_decimal(u[0])
                sa, sb, profit = calc_bets(TOTAL_STAKE, w_o, w_u, dec_o, dec_u)
                arbs.append({
                    "type": "total",
                    "market": f"{away} @ {home} TOTAL {pt}",
                    "side_a": f"Over @ {o[1]} ({fmt_odds(o[0])})",
                    "side_b": f"Under @ {u[1]} ({fmt_odds(u[0])})",
                    "margin_%": round(margin, 3),
                    "stake_a": sa, "stake_b": sb, "profit": profit,
                })

        # spreads: pair home@+pt with away@-pt
        spread_points_home = {p for (m, n, p) in best if m == "spreads" and n == home}
        for pt in spread_points_home:
            h = best.get(("spreads", home, pt))
            a = best.get(("spreads", away, -pt)) if pt is not None else None
            if not h or not a:
                continue
            margin, w_h, w_a = check_arb(h[0], a[0])
            if margin > MIN_MARGIN:
                dec_h, dec_a = american_to_decimal(h[0]), american_to_decimal(a[0])
                sa, sb, profit = calc_bets(TOTAL_STAKE, w_h, w_a, dec_h, dec_a)
                arbs.append({
                    "type": "spread",
                    "market": f"{away} @ {home} SPREAD {pt:+}",
                    "side_a": f"{home} {pt:+} @ {h[1]} ({fmt_odds(h[0])})",
                    "side_b": f"{away} {-pt:+} @ {a[1]} ({fmt_odds(a[0])})",
                    "margin_%": round(margin, 3),
                    "stake_a": sa, "stake_b": sb, "profit": profit,
                })
    return arbs