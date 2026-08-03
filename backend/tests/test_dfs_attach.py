from app.schemas.wnba_props import WnbaPropBookQuote, WnbaPropLine
from app.services.dfs_attach import attach_dfs_snapshots


def _sb(player: str, market: str, side: str, **books) -> WnbaPropLine:
    return WnbaPropLine(
        player_name=player,
        stat=market.replace("player_", "").replace("_", " ").title(),
        market_type=market,
        side=side,
        **books,
    )


def test_drops_demon_and_keeps_standard_pp():
    props = attach_dfs_snapshots(
        [],
        [
            {"player_name": "A", "stat_type": "Points", "line_score": 19.5, "odds_type": "demon"},
            {"player_name": "A", "stat_type": "Points", "line_score": 18.5, "odds_type": "standard"},
        ],
        [],
    )
    assert len(props) == 2  # over+under
    assert all(p.prizepicks and p.prizepicks.line == 18.5 for p in props)
    assert all(p.prizepicks.odds_american == 100 for p in props)


def test_pp_only_and_ud_only_kept():
    pp = attach_dfs_snapshots([], [{"player_name": "A", "stat_type": "Points", "line_score": 10.5, "odds_type": "standard"}], [])
    assert {p.side for p in pp} == {"over", "under"}
    ud = attach_dfs_snapshots([], [], [{"player_name": "B", "stat_name": "assists", "line_score": 5.5, "side": "over", "american_price": -110}])
    assert len(ud) == 1 and ud[0].underdog.line == 5.5


def test_sportsbook_only_dropped():
    sb = [_sb("A", "player_points", "over", fanduel=WnbaPropBookQuote(line=20.5, odds_american=-110))]
    assert attach_dfs_snapshots(sb, [], []) == []


def test_exact_dfs_line_wins_over_farther_alt():
    sb = [
        _sb("Caitlin Clark", "player_points", "over",
            fanduel=WnbaPropBookQuote(line=19.5, odds_american=-110),
            draftkings=WnbaPropBookQuote(line=20.5, odds_american=-105)),
    ]
    out = attach_dfs_snapshots(
        sb,
        [{"player_name": "Caitlin Clark", "stat_type": "Points", "line_score": 19.5, "odds_type": "standard"}],
        [{"player_name": "Caitlin Clark", "stat_name": "points", "line_score": 19.5, "side": "over", "american_price": -108}],
    )
    over = next(p for p in out if p.side == "over")
    assert over.prizepicks.line == 19.5
    assert over.underdog.odds_american == -108
    assert over.fanduel.line == 19.5
    assert over.draftkings.line == 20.5


def test_prefers_exact_match_when_multiple_quotes_indexed():
    sb = [_sb("A", "player_points", "over", fanduel=WnbaPropBookQuote(line=20.5, odds_american=-110))]
    out = attach_dfs_snapshots(
        sb,
        [{"player_name": "A", "stat_type": "Points", "line_score": 19.5, "odds_type": "standard"}],
        [],
    )
    over = next(p for p in out if p.side == "over")
    assert over.fanduel.line == 20.5  # closest available


def test_parlay_dfs_quotes_ignored():
    sb = [_sb("A", "player_points", "over",
              prizepicks=WnbaPropBookQuote(line=99.0),
              underdog=WnbaPropBookQuote(line=99.0, odds_american=-100),
              fanduel=WnbaPropBookQuote(line=19.5, odds_american=-110))]
    out = attach_dfs_snapshots(
        sb,
        [{"player_name": "A", "stat_type": "Points", "line_score": 19.5, "odds_type": "standard"}],
        [],
    )
    over = next(p for p in out if p.side == "over")
    assert over.prizepicks.line == 19.5
    assert over.underdog is None


def test_pick_closest_prefers_exact_either_target():
    from app.services.dfs_attach import pick_closest_quote

    q19 = WnbaPropBookQuote(line=19.5, odds_american=-110)
    q21 = WnbaPropBookQuote(line=21.5, odds_american=-105)
    assert pick_closest_quote([q21, q19], [19.5, 20.5]) is q19


def test_differing_pp_and_ud_lines_get_separate_slots():
    sb = [
        _sb(
            "A'ja Wilson",
            "player_points",
            "over",
            fanduel=WnbaPropBookQuote(line=22.5, odds_american=-110),
            draftkings=WnbaPropBookQuote(line=23.5, odds_american=-105),
        ),
    ]
    out = attach_dfs_snapshots(
        sb,
        [
            {
                "player_name": "A'ja Wilson",
                "stat_type": "Points",
                "line_score": 22.5,
                "odds_type": "standard",
            }
        ],
        [
            {
                "player_name": "A'ja Wilson",
                "stat_name": "points",
                "line_score": 23.5,
                "side": "over",
                "american_price": -108,
            }
        ],
    )
    overs = [p for p in out if p.side == "over"]
    assert len(overs) == 2
    pp_slot = next(p for p in overs if p.prizepicks is not None)
    ud_slot = next(p for p in overs if p.underdog is not None)
    assert pp_slot.prizepicks.line == 22.5
    assert pp_slot.underdog is None
    assert pp_slot.fanduel and pp_slot.fanduel.line == 22.5
    assert ud_slot.underdog.line == 23.5
    assert ud_slot.prizepicks is None
    assert ud_slot.draftkings and ud_slot.draftkings.line == 23.5


def test_same_pp_and_ud_line_share_one_slot():
    out = attach_dfs_snapshots(
        [],
        [
            {
                "player_name": "A",
                "stat_type": "Points",
                "line_score": 19.5,
                "odds_type": "standard",
            }
        ],
        [
            {
                "player_name": "A",
                "stat_name": "points",
                "line_score": 19.5,
                "side": "over",
                "american_price": -108,
            }
        ],
    )
    overs = [p for p in out if p.side == "over"]
    assert len(overs) == 1
    assert overs[0].prizepicks.line == 19.5
    assert overs[0].underdog.line == 19.5


def test_underdog_alts_collapse_to_one_main_per_side():
    """Underdog snapshots include many alt lines; keep one main per player/stat/side."""
    out = attach_dfs_snapshots(
        [],
        [],
        [
            {
                "player_name": "A'ja Wilson",
                "stat_name": "points",
                "line_score": 34.5,
                "side": "over",
                "american_price": 250,
            },
            {
                "player_name": "A'ja Wilson",
                "stat_name": "points",
                "line_score": 22.5,
                "side": "over",
                "american_price": -110,
            },
            {
                "player_name": "A'ja Wilson",
                "stat_name": "points",
                "line_score": 18.5,
                "side": "over",
                "american_price": -180,
            },
        ],
    )
    overs = [p for p in out if p.side == "over"]
    assert len(overs) == 1
    assert overs[0].underdog.line == 22.5
    assert overs[0].underdog.odds_american == -110


def test_ud_main_prefers_line_matching_prizepicks():
    out = attach_dfs_snapshots(
        [],
        [
            {
                "player_name": "A'ja Wilson",
                "stat_type": "Points",
                "line_score": 22.5,
                "odds_type": "standard",
            }
        ],
        [
            {
                "player_name": "A'ja Wilson",
                "stat_name": "points",
                "line_score": 29.5,
                "side": "over",
                "american_price": -105,
            },
            {
                "player_name": "A'ja Wilson",
                "stat_name": "points",
                "line_score": 22.5,
                "side": "over",
                "american_price": 120,
            },
        ],
    )
    overs = [p for p in out if p.side == "over"]
    # Same line as PP → one shared slot (not a far UD alt as a second row).
    assert len(overs) == 1
    assert overs[0].prizepicks.line == 22.5
    assert overs[0].underdog.line == 22.5
