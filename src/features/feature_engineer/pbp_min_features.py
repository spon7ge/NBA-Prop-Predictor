import re
import ast
import pandas as pd
import numpy as np
from typing import Optional


_PERIOD_SECONDS = {**{p: 720 for p in range(1, 5)}}   # Q1–Q4 = 12 min
_OT_SECONDS = 300                                       # any OT period = 5 min

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def _clock_to_seconds(clock_str: str) -> float:
    """
    Convert NBA API clock string to seconds remaining in the period.
    'PT11M45.00S' → 705.0
    Returns 0.0 on parse failure.
    """
    m = _CLOCK_RE.match(str(clock_str))
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def _period_max_seconds(period: int) -> float:
    return float(_PERIOD_SECONDS.get(period, _OT_SECONDS))


def _parse_lineup(val) -> frozenset:
    """
    Accept tuple, list, frozenset, or stringified version of any of those.
    Returns frozenset[int].
    """
    if isinstance(val, (tuple, list, frozenset, set)):
        return frozenset(int(x) for x in val if x)
    if isinstance(val, str) and val:
        return frozenset(int(x) for x in ast.literal_eval(val))
    return frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# CORE STINT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def get_player_stints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts player-level stints from play-by-play data that has been
    processed by add_lineup_columns() (home5 / away5 columns).

    Clock is parsed directly from the `clock` column (PT##M##.##S format).
    No pre-computed MIN_SECONDS column is required.

    Returns one row per stint:
        personId      – player ID
        playerName    – display name
        teamId        – team
        period        – quarter / OT period
        entry_sec     – seconds remaining in period when player entered
        exit_sec      – seconds remaining in period when player exited
        stint_sec     – duration of this stint (seconds)
        stint_min     – duration of this stint (minutes, 4 dp)
        stint_num     – nth stint for this player in this period (1-indexed)
        is_starter    – True if player opened the period (entered at full period time)
        exit_reason   – 'substitution' | 'end_of_period' | 'end_of_game'
    """
    df = df.copy().reset_index(drop=True)

    # ── resolve home / away team IDs ─────────────────────────────────────────
    def _get_team_id(loc: str) -> int:
        ids = df.loc[(df["location"] == loc) & (df["teamId"] != 0), "teamId"].dropna()
        if ids.empty:
            raise ValueError(f"No team found for location='{loc}'")
        return int(ids.iloc[0])

    home_team_id = _get_team_id("h")
    away_team_id = _get_team_id("v")

    # ── personId → name lookup ────────────────────────────────────────────────
    pid_to_name: dict[int, str] = {}
    pid_to_team: dict[int, int] = {}
    for _, row in df.iterrows():
        pid = row.get("personId")
        tid = row.get("teamId")
        if pid and pd.notna(pid) and int(pid) != 0:
            pid = int(pid)
            name = row.get("playerNameI") or row.get("playerName")
            if pd.notna(name) and name:
                pid_to_name.setdefault(pid, str(name))
            if tid and tid in (home_team_id, away_team_id):
                pid_to_team.setdefault(pid, int(tid))

    # supplement team map from lineup columns (catches players with no solo action rows)
    for _, row in df.iterrows():
        for col, tid in (("home5", home_team_id), ("away5", away_team_id)):
            for pid in _parse_lineup(row.get(col)):
                pid_to_team.setdefault(pid, tid)

    # ── state ─────────────────────────────────────────────────────────────────
    stints: list[dict] = []
    active_entry: dict[int, float] = {}    # pid → seconds_remaining at entry
    stint_count: dict[tuple, int] = {}     # (pid, period) → count
    is_starter: dict[int, bool] = {}       # pid → opened this period?

    current_period: Optional[int] = None
    prev_home: frozenset = frozenset()
    prev_away: frozenset = frozenset()

    def _close_stint(pid: int, period: int, exit_sec: float, reason: str) -> None:
        entry_sec = active_entry.pop(pid, exit_sec)
        duration  = max(entry_sec - exit_sec, 0.0)
        key       = (pid, period)
        stint_count[key] = stint_count.get(key, 0) + 1
        stints.append({
            "personId":    pid,
            "playerName":  pid_to_name.get(pid, str(pid)),
            "teamId":      pid_to_team.get(pid),
            "period":      period,
            "entry_sec":   round(entry_sec, 2),
            "exit_sec":    round(exit_sec, 2),
            "stint_sec":   round(duration, 2),
            "stint_min":   round(duration / 60, 4),
            "stint_num":   stint_count[key],
            "is_starter":  is_starter.get(pid, False),
            "exit_reason": reason,
        })

    # ── row-by-row walk ───────────────────────────────────────────────────────
    n_rows = len(df)
    for i, row in df.iterrows():
        period      = int(row["period"])
        time_sec    = _clock_to_seconds(row["clock"])
        home_lineup = _parse_lineup(row.get("home5"))
        away_lineup = _parse_lineup(row.get("away5"))

        # ── period boundary ──────────────────────────────────────────────────
        if period != current_period:
            # Close all active stints from the previous period
            if current_period is not None:
                reason = "end_of_game" if i == n_rows - 1 else "end_of_period"
                for pid in list(active_entry):
                    _close_stint(pid, current_period, 0.0, reason)

            current_period = period
            start_sec = _period_max_seconds(period)
            active_entry.clear()
            is_starter.clear()

            # Seed starters from the opening lineup
            for pid in home_lineup | away_lineup:
                active_entry[pid] = start_sec
                is_starter[pid]   = True

            prev_home = home_lineup
            prev_away = away_lineup
            continue

        # ── detect lineup changes (substitutions) ─────────────────────────────
        home_out = prev_home - home_lineup
        home_in  = home_lineup - prev_home
        away_out = prev_away - away_lineup
        away_in  = away_lineup - prev_away

        for pid in home_out | away_out:
            if pid in active_entry:
                _close_stint(pid, period, time_sec, "substitution")

        for pid in home_in | away_in:
            active_entry[pid] = time_sec
            is_starter[pid]   = False

        prev_home = home_lineup
        prev_away = away_lineup

    # Close any stints still open at the final row
    if current_period is not None:
        for pid in list(active_entry):
            _close_stint(pid, current_period, 0.0, "end_of_game")

    return pd.DataFrame(stints)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY AGGREGATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_player_minutes_summary(stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player, per-period summary useful for minutes prediction modelling.

    Columns:
        personId, playerName, teamId, period,
        total_min         – total minutes played this period
        num_stints        – number of stints
        avg_stint_min     – average stint length
        max_stint_min     – longest single stint
        first_entry_sec   – seconds remaining on first entry (counts DOWN → higher = earlier)
        last_exit_sec     – seconds remaining on last exit
        is_starter        – started the period?
        pct_period_played – fraction of period on court (0–1)
    """
    period_max = stints_df["period"].map(lambda p: _period_max_seconds(p) / 60)

    grp = stints_df.groupby(["personId", "playerName", "teamId", "period"])
    summary = grp.agg(
        total_min       = ("stint_min", "sum"),
        num_stints      = ("stint_min", "count"),
        avg_stint_min   = ("stint_min", "mean"),
        max_stint_min   = ("stint_min", "max"),
        first_entry_sec = ("entry_sec", "max"),   # max because seconds count DOWN
        last_exit_sec   = ("exit_sec", "min"),
        is_starter      = ("is_starter", "any"),
    ).reset_index()

    # pct_period_played uses the correct max for Q vs OT
    summary["period_max_min"] = summary["period"].map(_period_max_seconds) / 60
    summary["pct_period_played"] = (
        (summary["total_min"] / summary["period_max_min"]).clip(0, 1).round(3)
    )
    summary = summary.drop(columns="period_max_min")

    summary["avg_stint_min"] = summary["avg_stint_min"].round(2)
    summary["total_min"]     = summary["total_min"].round(2)

    return summary.sort_values(["teamId", "period", "personId"]).reset_index(drop=True)


def get_game_minutes_summary(stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player game-level summary and predictive features.

    Columns:
        personId, playerName, teamId,
        total_game_min        – total game minutes
        periods_played        – distinct periods appeared in
        total_stints          – total stints in game
        avg_stint_min         – average stint length
        max_stint_min         – longest stint
        periods_started       – how many periods they opened
        first_period_on       – earliest period they appeared
        avg_entry_sec         – average seconds remaining when entering
                                (lower = tends to enter later in periods)
        entry_regularity_std  – std dev of entry times
                                (lower = more predictable rotation)
        first_stint_min       – duration of player's first Q1 stint (minutes)
        in_closing_lineup     – True if on court at any point in final 2 min
    """
    grp = stints_df.groupby(["personId", "playerName", "teamId"])
    summary = grp.agg(
        total_game_min       = ("stint_min", "sum"),
        total_stints         = ("stint_min", "count"),
        avg_stint_min        = ("stint_min", "mean"),
        max_stint_min        = ("stint_min", "max"),
        periods_played       = ("period", "nunique"),
        periods_started      = ("is_starter", "sum"),
        first_period_on      = ("period", "min"),
        avg_entry_sec        = ("entry_sec", "mean"),
        entry_regularity_std = ("entry_sec", "std"),
    ).reset_index()

    for col in ["total_game_min", "avg_stint_min", "max_stint_min",
                "avg_entry_sec", "entry_regularity_std"]:
        summary[col] = summary[col].round(2)

    # first_stint_min: duration of the player's opening stint in Q1
    q1_first = (
        stints_df[(stints_df["period"] == 1) & (stints_df["stint_num"] == 1)]
        [["personId", "stint_min"]]
        .rename(columns={"stint_min": "first_stint_min"})
    )
    summary = summary.merge(q1_first, on="personId", how="left")
    summary["first_stint_min"] = summary["first_stint_min"].round(2)

    # in_closing_lineup: player had any stint overlapping the final 2 min of the game
    # A stint covers seconds-remaining [exit_sec → entry_sec]; last 2 min = [0, 120].
    final_period = int(stints_df["period"].max())
    closing_pids = stints_df[
        (stints_df["period"] == final_period) &
        (stints_df["entry_sec"] > 0) &
        (stints_df["exit_sec"] < 120)
    ]["personId"].unique()
    summary["in_closing_lineup"] = summary["personId"].isin(closing_pids)

    return summary.sort_values(
        ["teamId", "total_game_min"], ascending=[True, False]
    ).reset_index(drop=True)


def get_clutch_minutes(pbp_df: pd.DataFrame, stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time each player spent on court during clutch windows: the last 5
    minutes of the final period when the score margin is 5 or fewer points.

    Returns DataFrame with columns [personId, clutch_min].
    Returns zeros for all players when score columns are absent.
    """
    home_col = next((c for c in ("scoreHome", "homeScore") if c in pbp_df.columns), None)
    away_col = next((c for c in ("scoreAway", "awayScore") if c in pbp_df.columns), None)
    all_pids = stints_df["personId"].unique()

    if home_col is None or away_col is None:
        return pd.DataFrame({"personId": all_pids, "clutch_min": 0.0})

    df = pbp_df.copy()
    df["_sec"]  = df["clock"].apply(_clock_to_seconds)
    df["_home"] = pd.to_numeric(df[home_col], errors="coerce").ffill()
    df["_away"] = pd.to_numeric(df[away_col], errors="coerce").ffill()
    df["_diff"] = (df["_home"] - df["_away"]).abs()

    final_period = int(stints_df["period"].max())
    period_rows = (
        df[df["period"] == final_period]
        .sort_values("_sec", ascending=False)
        .reset_index(drop=True)
    )

    # Each consecutive pair of PBP rows defines a time window; mark it clutch
    # if seconds remaining <= 300 and score margin <= 5 at the window's start.
    clutch_windows: list[tuple[float, float]] = []
    for i in range(len(period_rows) - 1):
        row = period_rows.iloc[i]
        nxt = period_rows.iloc[i + 1]
        w_hi, w_lo = float(row["_sec"]), float(nxt["_sec"])
        if w_hi <= 300.0 and float(row["_diff"]) <= 5.0:
            clutch_windows.append((w_hi, w_lo))
    if not period_rows.empty:
        last = period_rows.iloc[-1]
        if float(last["_sec"]) <= 300.0 and float(last["_diff"]) <= 5.0:
            clutch_windows.append((float(last["_sec"]), 0.0))

    final_stints = stints_df[stints_df["period"] == final_period]
    pid_clutch: dict[int, float] = {}

    for pid, group in final_stints.groupby("personId"):
        total = 0.0
        for _, stint in group.iterrows():
            s_hi = float(stint["entry_sec"])
            s_lo = float(stint["exit_sec"])
            for w_hi, w_lo in clutch_windows:
                overlap = max(0.0, min(s_hi, w_hi) - max(s_lo, w_lo))
                total += overlap
        pid_clutch[int(pid)] = round(total / 60, 4)

    rows = [{"personId": pid, "clutch_min": val} for pid, val in pid_clutch.items()]
    if not rows:
        return pd.DataFrame({"personId": all_pids, "clutch_min": 0.0})

    clutch_result = pd.DataFrame(rows)
    missing_pids = set(int(p) for p in all_pids) - set(pid_clutch.keys())
    if missing_pids:
        clutch_result = pd.concat([
            clutch_result,
            pd.DataFrame({"personId": list(missing_pids), "clutch_min": 0.0}),
        ], ignore_index=True)
    return clutch_result


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def analyze_player_minutes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    One-call wrapper. Input df must already have home5 / away5 columns
    from add_lineup_columns(). Returns:
        'stints'    – raw stint-level rows
        'by_period' – per-player per-period summary
        'by_game'   – per-player game totals + predictive features
                      (includes first_stint_min, in_closing_lineup,
                       clutch_min, clutch_min_rate)
    """
    stints    = get_player_stints(df)
    by_period = get_player_minutes_summary(stints)
    by_game   = get_game_minutes_summary(stints)

    clutch = get_clutch_minutes(df, stints)
    by_game = by_game.merge(clutch, on="personId", how="left")
    by_game["clutch_min"] = by_game["clutch_min"].fillna(0.0)
    by_game["clutch_min_rate"] = (
        by_game["clutch_min"] / by_game["total_game_min"].replace(0, np.nan)
    ).round(3).fillna(0.0)

    periods = sorted(stints["period"].unique())
    print("=" * 55)
    print(f"  Stints extracted  : {len(stints)}")
    print(f"  Players tracked   : {stints['personId'].nunique()}")
    print(f"  Periods covered   : {periods}")
    print("=" * 55)

    return {"stints": stints, "by_period": by_period, "by_game": by_game}