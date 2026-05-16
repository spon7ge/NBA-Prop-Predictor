import pandas as pd
import numpy as np
import isodate
import re
import unicodedata
from collections import defaultdict


# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

def _get_location(df: pd.DataFrame, player_id: int) -> str:
    """Return 'h' or 'v' for the given player."""
    row = df[df["personId"] == player_id]
    return row["location"].iloc[0] if not row.empty else "h"


def _score_margin(row: pd.Series, location: str) -> int:
    """Signed score margin from the player's team perspective."""
    h = int(row["scoreHome"])
    a = int(row["scoreAway"])
    return (h - a) if location == "h" else (a - h)

def cleanPlaybyPlay(df):
    df['MIN_SECONDS'] = df['clock'].apply(lambda x: isodate.parse_duration(x).total_seconds())
    df['scoreHome'] = df['scoreHome'].replace('', np.nan).ffill()
    df['scoreAway'] = df['scoreAway'].replace('', np.nan).ffill()
    return df

# ─────────────────────────────────────────────
# LINEUP TRACKING
# ─────────────────────────────────────────────

def build_global_name_map(full_df: pd.DataFrame) -> dict[str, int]:
    """
    Build a name→personId map from ALL games in the dataset.
    Pass the result into add_lineup_columns as `global_name_map` so players
    who never appear as personId in a specific game can still be resolved.
    """
    name_map: dict[str, int] = {}
    for _, row in full_df.iterrows():
        pid = row.get("personId")
        if not (pid and pid != 0 and pd.notna(pid)):
            continue
        pid = int(pid)
        for col in ("playerName", "playerNameI"):
            val = row.get(col)
            if pd.notna(val) and val:
                full = str(val).strip()
                for key in _name_variants(full):
                    if key not in name_map:
                        name_map[key] = pid
    return name_map


def _name_variants(full: str) -> list[str]:
    """All lookup keys we store for a given name string."""
    variants = [full, _ascii(full)]
    parts = full.split()
    if len(parts) > 1:
        last_part = " ".join(parts[1:])
        variants += [last_part, _ascii(last_part)]
    return variants

# ── Name normalization helpers ────────────────────────────────────────────────

def _ascii(s: str) -> str:
    """Strip diacritics and lowercase: 'Vučević' → 'vucevic'."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()


def _name_variants(name: str) -> list[str]:
    """
    All lookup keys to try for a given name string.
    Covers: raw, ascii-normalized, last-name-only, each hyphenated segment.
    """
    name = name.strip()
    variants = {name, _ascii(name)}
    parts = re.split(r"\s+", name)
    if len(parts) > 1:
        variants.add(parts[-1])
        variants.add(_ascii(parts[-1]))
    for part in re.split(r"[-\s]", name):
        if len(part) > 2:
            variants.add(part)
            variants.add(_ascii(part))
    return [v for v in variants if v]


def _parse_incoming_name(description: str) -> str | None:
    """Extract incoming player name from 'SUB: X FOR Y'."""
    m = re.match(r"SUB:\s+(.+?)\s+FOR\s+", str(description))
    return m.group(1).strip() if m else None


# ── Gameplay action types that reliably reveal who is on the court ────────────
_GAMEPLAY = frozenset({
    "Made Shot", "Missed Shot", "Rebound", "Foul",
    "Turnover", "Free Throw", "Violation",
})


# ── Main function ─────────────────────────────────────────────────────────────

def add_lineup_columns(
    df: pd.DataFrame,
    global_name_map: dict[tuple[int, str], int] | None = None,
) -> pd.DataFrame:
    """
    Adds `home5` and `away5` columns — tuples of 5 personIds currently on court.

    Approach 3 + 4:
      - personId on a sub row IS the player going OUT — used directly, no lookup.
      - Incoming player resolved via team-scoped (team_id, name_variant) → [personIds].
      - When a name maps to multiple candidates, bench filtering (roster − lineup)
        breaks the tie.
      - Period opener inference uses BOTH gameplay rows AND sub outgoing players:
          * Gameplay rows seen before the first sub for a team → guaranteed starters.
          * Sub outgoing players → guaranteed starters (they were on court).
          * Gameplay rows after a sub → starters only, excluding known substitutes.
        This correctly handles cases where a starter is subbed out before touching
        the ball (NBA API period-boundary data quirks).
      - If incoming resolves to a player already on court (data anomaly), the swap is
        skipped and the lineup held intact — no silent corruption to a sub-5 unit.
      - If out_id is not in the current lineup (opener inference missed them), add
        out_id first so the discard/add swap stays size-neutral at 5.
    """
    df = df.copy().reset_index(drop=True)

    def is_valid(pid) -> bool:
        try:
            return bool(pid and int(pid) != 0 and pd.notna(pid))
        except (TypeError, ValueError):
            return False

    # ── Resolve home / away team IDs from `location` ─────────────────────────
    def get_team_id(loc: str) -> int:
        ids = df.loc[(df["location"] == loc) & (df["teamId"] != 0), "teamId"].dropna()
        if ids.empty:
            raise ValueError(f"No team found for location='{loc}'")
        return int(ids.iloc[0])

    home_tid = get_team_id("h")
    away_tid = get_team_id("v")

    # ── Build full roster per team ────────────────────────────────────────────
    team_roster: dict[int, set[int]] = {home_tid: set(), away_tid: set()}
    for _, row in df.iterrows():
        pid, tid = row["personId"], row["teamId"]
        if is_valid(pid) and tid in team_roster:
            team_roster[tid].add(int(pid))

    # ── Build team-scoped name → candidate list lookup ────────────────────────
    # Key: (team_id, name_variant) → [personIds]  (usually one; >1 for same-last-name teammates)
    name_to_candidates: dict[tuple[int, str], list[int]] = defaultdict(list)

    for _, row in df.iterrows():
        pid, tid = row["personId"], row["teamId"]
        if not is_valid(pid) or tid not in team_roster:
            continue
        pid = int(pid)
        for col in ("playerName", "playerNameI"):
            val = row.get(col)
            if pd.notna(val) and val:
                for variant in _name_variants(str(val)):
                    key = (int(tid), variant)
                    if pid not in name_to_candidates[key]:
                        name_to_candidates[key].append(pid)

    def _quick_resolve(name: str, team_id: int) -> int | None:
        """Best-effort single-result resolve used only during opener inference."""
        for v in _name_variants(name):
            hits = name_to_candidates.get((team_id, v), [])
            if len(hits) == 1:
                return hits[0]
        return None

    # ── Incoming-player resolver (used during Pass 2 swap) ────────────────────
    def resolve_incoming(
        description: str,
        team_id: int,
        current_lineup: set[int],
    ) -> tuple[int | None, str]:
        """
        Returns (personId | None, reason).

        Variants are tried in descending specificity (longest string first).
        This handles same-last-name teammates on the same team, e.g. LeBron
        James (playerNameI='L. James') and Bronny James (playerNameI='B. James'):
        the variant 'L. James' uniquely identifies LeBron, while the bare
        'James' variant would match both. By short-circuiting on the first
        variant that yields a clean bench match, we never accumulate the
        ambiguous bare-last-name candidates.

        Resolution order per variant (most → least specific):
          1. Single bench candidate from this variant → return immediately.
          2. Single total candidate not on court → return immediately.
          3. Multiple bench candidates → continue to next (more specific) variant.
        After all variants exhausted:
          4. global_name_map fallback with same per-variant short-circuit.
          5. Failure → (None, reason).
        """
        incoming_name = _parse_incoming_name(description)
        if incoming_name is None:
            return None, "could not parse description"

        bench = team_roster[team_id] - current_lineup

        # Sort variants longest-first so the most specific (e.g. 'L. James')
        # is tried before the ambiguous bare last name (e.g. 'james').
        all_variants = sorted(_name_variants(incoming_name), key=len, reverse=True)

        last_candidates: list[int] = []   # kept for the final error message

        for variant in all_variants:
            hits = name_to_candidates.get((team_id, variant), [])
            if not hits:
                continue

            bench_hits = [p for p in hits if p in bench]

            if len(bench_hits) == 1:
                # Unambiguous bench match — best possible outcome.
                return bench_hits[0], "ok"

            if len(hits) == 1:
                pid = hits[0]
                if pid in current_lineup:
                    # Only one candidate but they're already on court.
                    # Keep trying less-specific variants before giving up —
                    # a less-specific variant might surface a bench player
                    # (shouldn't happen in clean data, but guards edge cases).
                    last_candidates = hits
                    continue
                return pid, "ok"

            # Multiple hits for this variant; record them and try next.
            last_candidates = hits

        # All variants exhausted — summarise why.
        all_candidates: list[int] = []
        for v in all_variants:
            for p in name_to_candidates.get((team_id, v), []):
                if p not in all_candidates:
                    all_candidates.append(p)

        if all_candidates:
            bench_cands = [p for p in all_candidates if p in bench]
            if not bench_cands:
                return None, f"all candidates {all_candidates} already on court — data anomaly"
            return None, f"ambiguous after all variants: bench candidates {bench_cands}"

        # --- global_name_map fallback ---
        if global_name_map:
            for variant in all_variants:
                pid = global_name_map.get((team_id, variant)) or global_name_map.get(variant)
                if pid is not None:
                    if pid in current_lineup:
                        return None, f"global map resolved to {pid} but already on court"
                    return pid, "ok (global map)"

        return None, f"'{incoming_name}' not in name lookup for team {team_id}"

    # ── Period opener inference ───────────────────────────────────────────────
    # Two signal types combined:
    #   Gameplay rows  → player was on court (starter or substitute post-sub)
    #   Sub outgoing   → player was definitely on court before this sub (= starter)
    #   Sub incoming   → player entered via sub (= NOT a starter; excluded from opener)
    #
    # Algorithm per team per period:
    #   Scan rows in time order. Maintain starters[] and known_subs set.
    #   Gameplay row: add pid to starters if not in known_subs and starters < 5.
    #   Sub row:      add out_id to starters (they were on court) if not already there
    #                 and starters < 5. Mark in_id as known_sub (not a starter).
    #                 If in_id was tentatively in starters, remove it.
    def infer_period_openers() -> dict[int, dict[int, list[int]]]:
        openers: dict[int, dict[int, list[int]]] = {}
        for period, period_df in df.groupby("period", sort=True):
            home_starters: list[int] = []
            away_starters: list[int] = []
            home_subs: set[int] = set()   # known substitutes — not starters
            away_subs: set[int] = set()

            for _, row in period_df.iterrows():
                action = row["actionType"]
                pid, tid = row["personId"], row["teamId"]

                if action in _GAMEPLAY and is_valid(pid):
                    pid = int(pid)
                    if tid == home_tid and pid not in home_subs and pid not in home_starters and len(home_starters) < 5:
                        home_starters.append(pid)
                    elif tid == away_tid and pid not in away_subs and pid not in away_starters and len(away_starters) < 5:
                        away_starters.append(pid)

                elif action == "Substitution" and is_valid(pid):
                    out_id = int(pid)
                    in_name = _parse_incoming_name(str(row.get("description", "")))
                    in_id = _quick_resolve(in_name, int(tid)) if in_name else None

                    if int(tid) == home_tid:
                        # outgoing = confirmed starter
                        if out_id not in home_starters and len(home_starters) < 5:
                            home_starters.append(out_id)
                        # incoming = known substitute (not a starter)
                        if in_id:
                            home_subs.add(in_id)
                            if in_id in home_starters:
                                home_starters.remove(in_id)
                    elif int(tid) == away_tid:
                        if out_id not in away_starters and len(away_starters) < 5:
                            away_starters.append(out_id)
                        if in_id:
                            away_subs.add(in_id)
                            if in_id in away_starters:
                                away_starters.remove(in_id)

                if len(home_starters) == 5 and len(away_starters) == 5:
                    break

            openers[period] = {home_tid: home_starters, away_tid: away_starters}
        return openers

    period_openers = infer_period_openers()

    # ── Pass 2: walk rows, apply subs, emit lineup snapshot per row ───────────
    home_lineup: set[int] = set()
    away_lineup: set[int] = set()
    current_period: int | None = None

    home5_out: list[tuple] = []
    away5_out: list[tuple] = []
    warnings: list[str] = []

    for _, row in df.iterrows():
        period  = int(row["period"])
        action  = row["actionType"]
        pid     = row["personId"]
        tid     = row["teamId"]
        desc    = row["description"]

        # ── Period boundary: seed from inferred opener ────────────────────────
        if period != current_period:
            opener        = period_openers[period]
            home_inferred = opener[home_tid]
            away_inferred = opener[away_tid]

            # Use inferred list as authoritative (5-player). If somehow short,
            # supplement with the carry-forward (players from prior period).
            if len(home_inferred) == 5:
                home_lineup = set(home_inferred)
            else:
                home_lineup = set(home_inferred) | {
                    p for p in home_lineup
                    if p not in home_inferred and len(home_inferred) + 1 <= 5
                }

            if len(away_inferred) == 5:
                away_lineup = set(away_inferred)
            else:
                away_lineup = set(away_inferred) | {
                    p for p in away_lineup
                    if p not in away_inferred and len(away_inferred) + 1 <= 5
                }

            current_period = period

        # ── Substitution ─────────────────────────────────────────────────────
        if action == "Substitution" and is_valid(pid):
            out_id = int(pid)
            lineup = home_lineup if int(tid) == home_tid else away_lineup
            side   = "home" if int(tid) == home_tid else "away"

            in_id, reason = resolve_incoming(desc, int(tid), lineup)

            if in_id is None:
                warnings.append(
                    f"Period {period} [{side}] '{desc}'"
                    f" | out={out_id} | {reason}"
                )
                # Do NOT touch lineup — hold it intact
            else:
                # If out_id isn't in the lineup, the opener inference missed that
                # player. Force them in first so the subsequent discard stays size-neutral.
                if out_id not in lineup and len(lineup) == 5:
                    # Remove the least-certain player: prefer to drop whoever came
                    # in as a substitute in this same period (known_subs), otherwise
                    # we can't know — just remove an arbitrary member and log it.
                    lineup.add(out_id)     # temporarily 6
                    # Drop in_id itself if it somehow snuck in (shouldn't happen),
                    # otherwise drop the player not seen in any gameplay action
                    ghost = next(
                        (p for p in lineup if p != out_id and p != in_id
                         and p not in {
                             int(r["personId"]) for _, r in df[
                                 (df["period"] == period) &
                                 (df["actionType"].isin(_GAMEPLAY)) &
                                 (df["teamId"] == int(tid))
                             ].iterrows() if is_valid(r["personId"])
                         }),
                        None
                    )
                    if ghost:
                        lineup.discard(ghost)
                    else:
                        # All 6 have gameplay evidence; drop whoever is oldest in
                        # the set by iterating (set order is insertion-ish in CPython)
                        lineup.discard(next(iter(lineup - {out_id, in_id})))

                lineup.discard(out_id)
                lineup.add(in_id)

        home5_out.append(tuple(sorted(home_lineup)))
        away5_out.append(tuple(sorted(away_lineup)))

    df["home5"] = home5_out
    df["away5"] = away5_out

    # ── Report ────────────────────────────────────────────────────────────────
    bad_home = sum(1 for t in home5_out if len(t) != 5)
    bad_away = sum(1 for t in away5_out if len(t) != 5)

    if warnings:
        print(f"\n⚠  {len(warnings)} substitution(s) skipped — lineup held at last known state:")
        for w in warnings:
            print(f"   {w}")
    if bad_home or bad_away:
        print(f"\n⚠  Rows with lineup != 5: home={bad_home}, away={bad_away}")
    if not warnings and not bad_home and not bad_away:
        print("✓ All substitutions resolved. Every row has exactly 5 per team.")

    return df

# ─────────────────────────────────────────────
# 1. MINUTES / STINTS
# ─────────────────────────────────────────────

def get_minutes_played(
    df: pd.DataFrame,
    player_name: str,
    team_id: int,
) -> dict:
    """
    Tracks substitution events for one player across a full game.

    Parameters
    ----------
    df          : play-by-play DataFrame for a single game
    player_name : playerName value as it appears in the data
    team_id     : team the player belongs to

    Returns
    -------
    total_minutes       : float   total minutes on court across full game
    stint_length_avg    : float   average stint length in seconds across full game
    first_sub_time      : float   seconds into Q1 before first exit (0 if never subbed out)
    n_stints            : int     total number of separate stints across full game
    quarters_played     : int     number of distinct quarters with court time
    """
    EMPTY = {
        "total_minutes":    0.0,
        "stint_length_avg": 0.0,
        "first_sub_time":   0,
        "n_stints":         0,
        "quarters_played":  0,
    }

    if df.empty:
        return EMPTY

    # ── Accumulate stints across all periods ──────────────────────────────────
    all_stint_lengths: list[float] = []
    first_sub_time: float | None   = None
    quarters_played: int           = 0

    for quarter in sorted(df["period"].unique()):
        QUARTER_START = 720 if quarter <= 4 else 300
        QUARTER_END   = 0

        quarter_df = df[df["period"] == quarter]

        # Check player appeared this quarter
        player_in_quarter = quarter_df[
            (quarter_df["teamId"]     == team_id) &
            (quarter_df["playerName"] == player_name)
        ]

        # Pull this team's subs in this quarter
        team_subs = (
            quarter_df[
                (quarter_df["teamId"]     == team_id) &
                (quarter_df["actionType"] == "Substitution")
            ]
            .sort_values("MIN_SECONDS")
        )

        # Build ordered (time, in/out) events for this player
        player_events: list[tuple[float, str]] = []
        for _, sub in team_subs.iterrows():
            desc = sub["description"]
            t    = sub["MIN_SECONDS"]
            if desc.startswith(f"SUB: {player_name} FOR "):
                player_events.append((t, "in"))
            elif f"FOR {player_name}" in desc:
                player_events.append((t, "out"))

        # Sort descending → forward game-time order (720 → 0)
        player_events.sort(key=lambda x: x[0], reverse=True)

        # ── Compute stint lengths for this quarter ────────────────────────────
        quarter_stints: list[float] = []

        if not player_events:
            if not player_in_quarter.empty:
                quarter_stints = [float(QUARTER_START)]
        else:
            on_court    = player_events[0][1] == "out"
            stint_start = QUARTER_START if on_court else None

            for t, etype in player_events:
                if etype == "in" and not on_court:
                    on_court    = True
                    stint_start = t

                elif etype == "out" and on_court and stint_start is not None:
                    duration = stint_start - t
                    quarter_stints.append(duration)

                    # Only record first_sub_time from Q1
                    if first_sub_time is None and quarter == 1:
                        first_sub_time = QUARTER_START - t

                    on_court    = False
                    stint_start = None

            if on_court and stint_start is not None:
                quarter_stints.append(stint_start - QUARTER_END)

        if quarter_stints:
            quarters_played      += 1
            all_stint_lengths.extend(quarter_stints)

    # ── Aggregate across full game ────────────────────────────────────────────
    if not all_stint_lengths:
        return EMPTY

    total_s = sum(all_stint_lengths)

    return {
        "total_minutes":    round(total_s / 60, 2),
        "stint_length_avg": round(total_s / len(all_stint_lengths), 2),
        "first_sub_time":   first_sub_time if first_sub_time is not None else 0,
        "n_stints":         len(all_stint_lengths),
        "quarters_played":  quarters_played,
    }

# ─────────────────────────────────────────────
# 2. GAME CONTEXT  (score margin, clutch, blowout)
# ─────────────────────────────────────────────

def get_game_context(
    full_quarter_df: pd.DataFrame,
    location: str,
) -> dict:
    """
    Quarter-level game state from the player's team perspective.

    Returns score margins, leading/trailing flags, clutch & blowout flags.
    """
    if full_quarter_df.empty:
        return {
            "score_margin_start": 0, "score_margin_end": 0,
            "score_margin_change": 0, "leading_team": 0,
            "trailing_team": 0, "clutch": 0, "blowout": 0,
        }

    start = _score_margin(full_quarter_df.iloc[0],  location)
    end   = _score_margin(full_quarter_df.iloc[-1], location)

    return {
        "score_margin_start":  start,
        "score_margin_end":    end,
        "score_margin_change": abs(end) - abs(start),
        "leading_team":        int(end > 0),
        "trailing_team":       int(end < 0),
        "clutch":              int(abs(end) <= 5),
        "blowout":             int(abs(end) >= 15),
    }


# ─────────────────────────────────────────────
# 3. SHOOTING  (FG, 3PT, FT, points)
# ─────────────────────────────────────────────

def get_shooting_stats(
    player_quarter_df: pd.DataFrame,
    minutes_played: float,
) -> dict:
    """
    All shooting volume and efficiency for one player in one quarter.
    """
    fg2m = fg2a = fg3m = fg3a = 0
    FTM = FTA = 0

    for _, row in player_quarter_df.iterrows():
        if row["isFieldGoal"] == 1:
            is3 = "3PT" in str(row["description"]).split()
            made = row["shotResult"] == "Made"
            if is3:
                fg3a += 1
                fg3m += int(made)
            else:
                fg2a += 1
                fg2m += int(made)

        if row["actionType"] == "Free Throw":
            FTA += 1
            FTM += int("MISS" not in str(row["description"]))

    fga   = fg2a + fg3a
    fgm   = fg2m + fg3m
    pts   = 2 * fg2m + 3 * fg3m + FTM
    min_  = minutes_played or 1  # avoid /0

    return {
        "points":         pts,
        "points_per_min": round(pts / min_, 2) if minutes_played else 0,
        "FGM":  fgm,  "FGA":  fga,
        "FG3M": fg3m, "FG3A": fg3a,
        "FTM":  FTM,  "FTA":  FTA,
        "FG_pct":  round(fgm  / fga,  2) if fga  else 0,
        "FG3_pct": round(fg3m / fg3a, 2) if fg3a else 0,
        "FT_pct":  round(FTM  / FTA,  2) if FTA  else 0,
        "eFG_pct": round((fgm + 0.5 * fg3m) / fga, 2) if fga else 0,
        "TS_pct":  round((fgm * 2 + fg3m + FTM) / (2 * (fga + 0.44 * FTA)), 2)
                   if (fga + 0.44 * FTA) else 0,
        "FGA_per_min":  round(fga  / min_, 2) if minutes_played else 0,
        "FG3A_per_min": round(fg3a / min_, 2) if minutes_played else 0,
        "FTA_per_min":  round(FTA  / min_, 2) if minutes_played else 0,
    }


# ─────────────────────────────────────────────
# 4. SHOT GEOGRAPHY  (close, mid-range, distance)
# ─────────────────────────────────────────────

def get_shot_geography(player_quarter_df: pd.DataFrame) -> dict:
    """
    Shot zone breakdown and average distances.
    """
    close_m = close_a = mid_m = mid_a = 0
    missed_dists: list[float] = []
    made_dists:   list[float] = []

    for _, row in player_quarter_df.iterrows():
        if row["isFieldGoal"] != 1:
            continue
        dist = row["shotDistance"]
        made = row["shotResult"] == "Made"

        (made_dists if made else missed_dists).append(dist)

        if dist <= 9:
            close_a += 1
            close_m += int(made)
        elif 10 <= dist <= 22:
            mid_a += 1
            mid_m += int(made)

    all_dists = missed_dists + made_dists

    return {
        "close_shots_made":      close_m,
        "close_shots_attempted": close_a,
        "close_shots_pct":       round(close_m / close_a, 2) if close_a else 0,
        "mid_range_made":        mid_m,
        "mid_range_attempted":   mid_a,
        "mid_range_pct":         round(mid_m / mid_a, 2) if mid_a else 0,
        "fg_made_dist_avg":      round(np.mean(made_dists),   2) if made_dists   else 0,
        "fg_missed_dist_avg":    round(np.mean(missed_dists), 2) if missed_dists else 0,
        "overall_dist_avg":      round(np.mean(all_dists),    2) if all_dists    else 0,
    }


# ─────────────────────────────────────────────
# 5. USAGE / ROLE
# ─────────────────────────────────────────────

def get_usage(
    player_quarter_df: pd.DataFrame,
    team_quarter_df:   pd.DataFrame,
    minutes_played:    float,
    fga: int, fta: int, turnovers: int,
) -> dict:
    """
    USG% and possession share — how central is this player to the offence?
    """
    QUARTER_MINUTES = 12.0

    team_fga = len(team_quarter_df[team_quarter_df["isFieldGoal"] == 1])
    team_fta = len(team_quarter_df[team_quarter_df["actionType"] == "Free Throw"])
    team_to  = len(team_quarter_df[team_quarter_df["actionType"] == "Turnover"])

    team_poss = team_fga + 0.44 * team_fta + team_to
    denom     = minutes_played * team_poss

    usg_pct = (
        100 * (fga + 0.44 * fta + turnovers) * QUARTER_MINUTES / denom
        if denom > 0 else 0
    )

    # Broader possession-share (all action types)
    action_types = ["Made Shot", "Missed Shot", "Free Throw",
                    "Turnover", "Rebound", "Foul", "Substitution"]
    player_actions = len(player_quarter_df[
        player_quarter_df["actionType"].isin(action_types)
    ])
    team_actions = len(team_quarter_df[
        team_quarter_df["actionType"].isin(action_types)
    ])

    return {
        "USG_pct":               round(usg_pct, 2),
        "player_possession_share": round(player_actions / team_actions, 3)
                                   if team_actions else 0,
        "total_actions":         player_actions,
        "total_actions_per_min": round(player_actions / minutes_played, 2)
                                  if minutes_played else 0,
    }


# ─────────────────────────────────────────────
# 6. SCORING RHYTHM  (streaks, bursts)
# ─────────────────────────────────────────────

def get_scoring_rhythm(player_quarter_df: pd.DataFrame) -> dict:
    """
    Hot/cold streaks and burst scoring windows.
    """
    fg_attempts = (
        player_quarter_df[player_quarter_df["isFieldGoal"] == 1]
        .sort_values("MIN_SECONDS")
        .reset_index(drop=True)
    )

    make_streak = miss_streak = make_max = miss_max = 0
    for _, row in fg_attempts.iterrows():
        if row["shotResult"] == "Made":
            make_streak += 1; make_max = max(make_max, make_streak); miss_streak = 0
        else:
            miss_streak += 1; miss_max = max(miss_max, miss_streak); make_streak = 0

    # Max points in any 60-second window
    points_in_burst = 0
    for i in range(len(fg_attempts)):
        t0   = fg_attempts.iloc[i]["MIN_SECONDS"]
        pts  = sum(
            int(r["shotValue"])
            for _, r in fg_attempts.iterrows()
            if r["MIN_SECONDS"] <= t0 + 60
            and r["shotResult"] == "Made"
            and pd.notna(r["shotValue"])
        )
        points_in_burst = max(points_in_burst, pts)

    return {
        "make_streak":     make_max,
        "miss_streak":     miss_max,
        "points_in_burst": points_in_burst,
    }


# ─────────────────────────────────────────────
# 7. TRANSITION / OPPONENT TURNOVERS
# ─────────────────────────────────────────────

def get_transition_scoring(
    player_quarter_df: pd.DataFrame,
    opponent_quarter_df: pd.DataFrame,
) -> dict:
    """
    Points scored in the immediate possession after an opponent turnover.
    """
    opp_tos = (
        opponent_quarter_df[opponent_quarter_df["actionType"] == "Turnover"]
        .sort_values("MIN_SECONDS")
    )

    made_shots = (
        player_quarter_df[
            (player_quarter_df["isFieldGoal"] == 1) &
            (player_quarter_df["shotResult"] == "Made")
        ]
        .sort_values("MIN_SECONDS")
        .reset_index(drop=True)
    )
    made_fts = (
        player_quarter_df[
            (player_quarter_df["actionType"] == "Free Throw") &
            (~player_quarter_df["description"].str.contains("MISS", na=False))
        ]
        .sort_values("MIN_SECONDS")
        .reset_index(drop=True)
    )

    pts_after_to = 0
    for _, to_row in opp_tos.iterrows():
        t = to_row["MIN_SECONDS"]
        # First made shot after the turnover
        next_shot = made_shots[made_shots["MIN_SECONDS"] > t]
        if not next_shot.empty:
            val = next_shot.iloc[0]["shotValue"]
            pts_after_to += int(val) if pd.notna(val) else 0
        # First made FT after the turnover
        next_ft = made_fts[made_fts["MIN_SECONDS"] > t]
        if not next_ft.empty:
            pts_after_to += 1

    return {"points_after_turnover": pts_after_to}


# ─────────────────────────────────────────────
# 8. MOMENTUM & GAME STATE RESPONSE
# ─────────────────────────────────────────────

def get_momentum_features(
    player_quarter_df: pd.DataFrame,
    full_quarter_df:   pd.DataFrame,
    location: str,
) -> dict:
    """
    Average margin when scoring, max momentum run, lead-change involvement.
    """
    full_sorted = full_quarter_df.sort_values("MIN_SECONDS")

    def margin_at(t: float) -> int:
        rows = full_sorted[full_sorted["MIN_SECONDS"] <= t]
        return _score_margin(rows.iloc[-1], location) if not rows.empty else 0

    # Average margin when this player scored
    scoring_events = pd.concat([
        player_quarter_df[
            (player_quarter_df["isFieldGoal"] == 1) &
            (player_quarter_df["shotResult"] == "Made")
        ],
        player_quarter_df[
            (player_quarter_df["actionType"] == "Free Throw") &
            (~player_quarter_df["description"].str.contains("MISS", na=False))
        ],
    ])
    margins = [margin_at(t) for t in scoring_events["MIN_SECONDS"]]
    margin_when_scoring_avg = round(np.mean(margins), 2) if margins else 0

    # Largest positive margin run across the quarter
    all_margins = [_score_margin(r, location) for _, r in full_sorted.iterrows()]
    momentum_run = 0
    if all_margins:
        low = all_margins[0]
        for m in all_margins[1:]:
            momentum_run = max(momentum_run, m - low)
            low = min(low, m)

    # Lead-change involvement
    lead_change_involvement = 0
    prev_leader = None
    for _, row in full_sorted.iterrows():
        m = _score_margin(row, location)
        leader = "us" if m > 0 else ("them" if m < 0 else "tied")
        if prev_leader and leader != prev_leader and leader != "tied":
            t = row["MIN_SECONDS"]
            nearby = player_quarter_df[
                (player_quarter_df["MIN_SECONDS"].between(t - 5, t + 5)) &
                (
                    ((player_quarter_df["isFieldGoal"] == 1) & (player_quarter_df["shotResult"] == "Made")) |
                    (player_quarter_df["actionType"] == "Assist") |
                    ((player_quarter_df["actionType"] == "Free Throw") &
                     (~player_quarter_df["description"].str.contains("MISS", na=False)))
                )
            ]
            if not nearby.empty:
                lead_change_involvement = 1
                break
        prev_leader = leader

    return {
        "margin_when_scoring_avg": margin_when_scoring_avg,
        "momentum_run":            momentum_run,
        "lead_change_involvement": lead_change_involvement,
    }


# ─────────────────────────────────────────────
# 9. TEMPORAL DISTRIBUTION
# ─────────────────────────────────────────────

def get_temporal_distribution(
    player_quarter_df: pd.DataFrame,
    full_quarter_df:   pd.DataFrame,
    location: str,
) -> dict:
    """
    Early-quarter vs late-quarter scoring share, and clutch FG%.
    """
    EARLY_CUTOFF = 540   # first 3 min  (720 → 540)
    LATE_CUTOFF  = 120   # final 2 min  (120 → 0)

    made_shots = player_quarter_df[
        (player_quarter_df["isFieldGoal"] == 1) &
        (player_quarter_df["shotResult"] == "Made")
    ]
    made_fts = player_quarter_df[
        (player_quarter_df["actionType"] == "Free Throw") &
        (~player_quarter_df["description"].str.contains("MISS", na=False))
    ]

    def shot_pts(df): return df["shotValue"].dropna().astype(int).sum()

    early_pts = (
        shot_pts(made_shots[made_shots["MIN_SECONDS"] >= EARLY_CUTOFF]) +
        len(made_fts[made_fts["MIN_SECONDS"] >= EARLY_CUTOFF])
    )
    late_pts = (
        shot_pts(made_shots[made_shots["MIN_SECONDS"] <= LATE_CUTOFF]) +
        len(made_fts[made_fts["MIN_SECONDS"] <= LATE_CUTOFF])
    )
    total_pts = shot_pts(made_shots) + len(made_fts)

    # Clutch FG% (final 2 min, margin ≤ 5)
    full_sorted = full_quarter_df.sort_values("MIN_SECONDS")
    clutch_fga = clutch_fgm = 0
    for _, shot in player_quarter_df[
        (player_quarter_df["isFieldGoal"] == 1) &
        (player_quarter_df["MIN_SECONDS"] <= LATE_CUTOFF)
    ].iterrows():
        nearby = full_sorted[full_sorted["MIN_SECONDS"] <= shot["MIN_SECONDS"]]
        if not nearby.empty:
            margin = abs(_score_margin(nearby.iloc[-1], location))
            if margin <= 5:
                clutch_fga += 1
                clutch_fgm += int(shot["shotResult"] == "Made")

    return {
        "early_points_share": round(early_pts / total_pts, 3) if total_pts else 0,
        "late_points_share":  round(late_pts  / total_pts, 3) if total_pts else 0,
        "clutch_fg_pct":      round(clutch_fgm / clutch_fga, 2) if clutch_fga else 0,
    }


# ─────────────────────────────────────────────
# 10. OTHER BOX STATS
# ─────────────────────────────────────────────

def get_other_stats(player_quarter_df: pd.DataFrame) -> dict:
    """Assists, turnovers, rebounds, steals, blocks, fouls."""
    desc = player_quarter_df["description"].fillna("")
    return {
        "assists":    (player_quarter_df["actionType"] == "Assist").sum(),
        "turnovers":  (player_quarter_df["actionType"] == "Turnover").sum(),
        "off_reb":    desc.str.contains("Off", na=False).sum(),
        "def_reb":    desc.str.contains("Def", na=False).sum(),
        "steals":     desc.str.contains("Steal", na=False).sum(),
        "blocks":     desc.str.contains("Block", na=False).sum(),
        "fouls":      (player_quarter_df["actionType"] == "Foul").sum(),
        "early_fouls":(
            player_quarter_df[player_quarter_df["actionType"] == "Foul"]
            ["MIN_SECONDS"].gt(600).sum()
        ),
    }