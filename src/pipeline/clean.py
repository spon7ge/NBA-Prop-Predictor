"""Silver layer: merge five ``raw.*`` game-log tables → ``silver.*``.

Reads from Supabase ``raw.nba_*`` / ``raw.wnba_*``, merges player + team +
tracking frames, drops ranks / duplicate / redundant columns, and upserts into
``silver.nba_player_gamelogs`` or ``silver.wnba_player_gamelogs``.

Also enriches each row with:
* ``pos``          — from the season positions CSV (then tracking fallback):
                     NBA keeps fine slots only (PG / SG / SF / PF / C);
                     WNBA keeps Basketball-Reference labels (G / F / C and
                     hybrids like G-F / F-C).
* ``game_total``   — Rotowire game over/under (NBA only).
* ``team_spread``  — Rotowire home-team spread (NBA only).

Rotowire CSVs are produced by ``src.scrapers.rotowire_scraper`` which uses an
API ``season`` string equal to the *start* year of the season, e.g. ``"2025"``
for the 2025-26 NBA season.  Expected path::

    data/raw/rotowire/rotowire_nba_{year}.csv

When the CSV is absent and ``auto_scrape_rotowire=True`` is passed to
``build_silver`` (or ``auto_scrape=True`` to ``load_rotowire``), the scraper
is invoked automatically via ``src.scrapers.rotowire_scraper.run_scrape``.

Player-position CSVs live in ``data/raw/player_positions/``.  Season selects
the file:

* NBA  — end year (``\"2025-26\"`` → ``nba_2026_players.csv``);
         ``player_positions.csv`` is the fallback when that file is missing.
* WNBA — calendar year (``\"2018\"`` → ``wnba_2018_players.csv``).

Matching uses ``name_s26`` when present, plus accent-stripped ``name``.

Examples::

    from src.pipeline.clean import build_silver, fetch_and_build_silver

    # Clean only (raw.* already loaded)
    build_silver("2025-26", "Regular Season", league="nba")

    # One shot: fetch endpoints → merge → pos + Rotowire → silver upsert
    fetch_and_build_silver("2025-26", "Regular Season", league="nba")

CLI::

    python -m src.pipeline.clean --league nba --season 2025-26
    python -m src.pipeline.clean --league wnba --season 2026 --auto-scrape-rotowire
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Literal

import pandas as pd

from src.utils.db import _normalize_col, read_df, upsert_df
from src.pipeline.fetch import LEAGUES, LeagueKey

# ---------------------------------------------------------------------------
# Paths / lookup maps
# ---------------------------------------------------------------------------

_ROTOWIRE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "rotowire"
_PLAYER_POSITIONS_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "player_positions"
)

# Rotowire sometimes uses different abbreviations than the NBA stats API.
_RW_ABBREV_MAP: dict[str, str] = {
    "PHO": "PHX",
    "GS": "GSW",
    "SA": "SAS",
    "NO": "NOP",
    "NY": "NYK",
    "UTAH": "UTA",
    "WSH": "WAS",
    "CHA": "CHO",
}

# Maps values that may appear in the tracking ``position`` field to a concise
# canonical form. NBA silver ``pos`` keeps only the five fine slots; WNBA
# keeps the coarser Basketball-Reference labels the scraper writes.
_FINE_POS: frozenset[str] = frozenset({"PG", "SG", "SF", "PF", "C"})
_WNBA_POS: frozenset[str] = frozenset(
    {"G", "F", "C", "G-F", "F-G", "F-C", "C-F"}
) | _FINE_POS
_START_POS_CANONICAL: dict[str, str] = {
    "Point Guard": "PG",
    "Shooting Guard": "SG",
    "Small Forward": "SF",
    "Power Forward": "PF",
    "Center": "C",
    "Guard": "G",
    "Forward": "F",
    "Guard-Forward": "G-F",
    "Forward-Guard": "F-G",
    "Forward-Center": "F-C",
    "Center-Forward": "C-F",
    "PG": "PG",
    "SG": "SG",
    "SF": "SF",
    "PF": "PF",
    "C": "C",
    "G": "G",
    "F": "F",
    "G-F": "G-F",
    "F-G": "F-G",
    "F-C": "F-C",
    "C-F": "C-F",
}

# BoxScorePlayerTrackV3 snake_case → abbreviated silver column names.
_TRACKING_SILVER_RENAME: dict[str, str] = {
    "position": "start_position",
    "speed": "spd",
    "distance": "dist",
    "rebound_chances_offensive": "orbc",
    "rebound_chances_defensive": "drbc",
    "rebound_chances_total": "rbc",
    "touches": "tchs",
    "secondary_assists": "sast",
    "free_throw_assists": "ftast",
    "passes": "pass",
    "contested_field_goals_made": "cfgm",
    "contested_field_goals_attempted": "cfga",
    "contested_field_goal_percentage": "cfg_pct",
    "uncontested_field_goals_made": "ufgm",
    "uncontested_field_goals_attempted": "ufga",
    "uncontested_field_goals_percentage": "ufg_pct",
    "defended_at_rim_field_goals_made": "dfgm",
    "defended_at_rim_field_goals_attempted": "dfga",
    "defended_at_rim_field_goal_percentage": "dfg_pct",
}

_SILVER_TABLE: dict[LeagueKey, str] = {
    "nba": "nba_player_gamelogs",
    "wnba": "wnba_player_gamelogs",
}

_DATASET_KEYS: tuple[str, ...] = (
    "player_base",
    "player_adv",
    "team_base",
    "team_adv",
    "start_positions",
)


def _nba_season_to_rotowire_year(season: str) -> str:
    """``'2025-26'`` → ``'2025'`` (Rotowire API season string)."""
    return season.split("-")[0]


def load_rotowire(
    season: str,
    *,
    league: LeagueKey = "nba",
    auto_scrape: bool = False,
) -> pd.DataFrame:
    """Load the pre-scraped Rotowire CSV for *season* (e.g. ``'2025-26'``).

    Returns an empty DataFrame for non-NBA leagues or when the file is absent.
    Columns returned: ``rw_date``, ``rw_away``, ``rw_home``,
    ``game_total``, ``team_spread`` (silver schema names).

    Parameters
    ----------
    auto_scrape:
        When True and the CSV is missing, run ``rotowire_scraper.run_scrape``
        automatically to fetch the data before loading.  Requires ``playwright``
        to be installed (``pip install playwright && playwright install chromium``).
    """
    if league != "nba":
        return pd.DataFrame()
    year = _nba_season_to_rotowire_year(season)
    path = _ROTOWIRE_DIR / f"rotowire_nba_{year}.csv"
    if not path.exists():
        if auto_scrape:
            print(f"  Rotowire CSV not found — scraping season {year} …")
            try:
                import asyncio
                from src.scrapers.rotowire_scraper import run_scrape
                asyncio.run(run_scrape(season=year, output_file=path))
            except Exception as exc:
                print(f"  ⚠ Rotowire auto-scrape failed: {exc}  — skipping enrichment")
                return pd.DataFrame()
        else:
            print(f"  ⚠ Rotowire CSV not found: {path}  — skipping enrichment")
            return pd.DataFrame()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Game" not in df.columns or "Tipoff" not in df.columns:
        print("  ⚠ Rotowire CSV missing expected columns (Game, Tipoff)")
        return pd.DataFrame()

    game_split = df["Game"].str.split(r"\s*@\s*", n=1, expand=True, regex=True)
    df["rw_away"] = game_split[0].str.strip().map(lambda x: _RW_ABBREV_MAP.get(x, x))
    df["rw_home"] = game_split[1].str.strip().map(lambda x: _RW_ABBREV_MAP.get(x, x))

    # Tipoff is "Oct 21 7:30 PM" (no year). Prefer the CSV Season column, then
    # bump Jan–Jul into the next calendar year (NBA season spans two years).
    season_year = (
        pd.to_numeric(df["Season"], errors="coerce").fillna(int(year)).astype(int)
        if "Season" in df.columns
        else pd.Series(int(year), index=df.index)
    )
    tip = df["Tipoff"].astype(str).str.strip()
    parsed = pd.to_datetime(tip + " " + season_year.astype(str), errors="coerce")
    jan_jul = parsed.dt.month.fillna(0).astype(int).between(1, 7)
    if jan_jul.any():
        parsed = parsed.copy()
        parsed.loc[jan_jul] = pd.to_datetime(
            tip[jan_jul] + " " + (season_year[jan_jul] + 1).astype(str),
            errors="coerce",
        )
    df["rw_date"] = parsed.dt.normalize()

    return (
        df[["rw_date", "rw_away", "rw_home", "Over_Under", "Home_Line"]]
        .rename(columns={"Over_Under": "game_total", "Home_Line": "team_spread"})
    )


def enrich_rotowire(df: pd.DataFrame, rw: pd.DataFrame) -> pd.DataFrame:
    """Left-join Rotowire O/U and line onto a silver player-game frame.

    Matches on game date + home/away team abbreviations parsed from ``matchup``
    (``"BOS @ MIA"`` → away=BOS, home=MIA; ``"BOS vs. MIA"`` → home=BOS, away=MIA).

    Writes silver columns ``game_total`` (over/under) and ``team_spread``
    (home-team line).
    """
    if rw.empty or df.empty:
        return df

    out = _snake_columns(df)
    if "game_date" not in out.columns or "matchup" not in out.columns:
        print("  ⚠ missing game_date/matchup — skipping Rotowire enrichment")
        return df

    # Both sides as normalized datetime64 so merge dtypes always agree.
    out["_rw_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    rw = rw.copy()
    rw["rw_date"] = pd.to_datetime(rw["rw_date"], errors="coerce").dt.normalize()

    def _parse_matchup(matchup: str) -> tuple[str | None, str | None]:
        m = str(matchup)
        if " @ " in m:
            away, home = m.split(" @ ", 1)
            return away.strip(), home.strip()
        if " vs. " in m:
            home, away = m.split(" vs. ", 1)
            return away.strip(), home.strip()
        return None, None

    parsed = out["matchup"].apply(_parse_matchup)
    out["_rw_away"] = parsed.apply(lambda t: _RW_ABBREV_MAP.get(t[0], t[0]) if t[0] else None)
    out["_rw_home"] = parsed.apply(lambda t: _RW_ABBREV_MAP.get(t[1], t[1]) if t[1] else None)

    merged = out.merge(
        rw,
        left_on=["_rw_date", "_rw_away", "_rw_home"],
        right_on=["rw_date", "rw_away", "rw_home"],
        how="left",
    ).drop(columns=["_rw_date", "_rw_away", "_rw_home", "rw_date", "rw_away", "rw_home"])

    matched = merged["game_total"].notna().sum()
    print(f"  Rotowire: {matched:,} / {len(merged):,} rows matched (game_total / team_spread)")
    return merged


def _nba_season_end_year(season: str) -> str:
    """``'2025-26'`` → ``'2026'`` (season-file suffix in ``nba_{end}_players.csv``)."""
    start_year = _nba_season_to_rotowire_year(season)
    return str(int(start_year) + 1)


def _wnba_season_year(season: str) -> str:
    """``'2018'`` / ``'2026'`` → calendar year for ``wnba_{year}_players.csv``."""
    s = str(season).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return s


def _norm_player_name(name: object) -> str:
    """Lowercase + strip accents so CSV ``name`` matches NBA API ``player_name``."""
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(n.lower().strip().split())


def load_player_positions(
    season: str | None = None,
    *,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    """Load player→position lookup from the ``data/raw/player_positions/`` CSVs.

    Picks the season file from the chosen season string:

    * NBA  ``\"2025-26\"`` → ``nba_2026_players.csv``
    * WNBA ``\"2018\"``    → ``wnba_2018_players.csv``

    For NBA, falls back to ``player_positions.csv`` when the season file is missing.

    Match keys prefer ASCII ``name_s26`` (when present), then accent-stripped
    ``name``, so API names like ``Luka Doncic`` hit CSV rows like ``Luka Dončić``.

    Returns a DataFrame with columns ``_name_lower`` and ``pos_csv``, or an
    empty DataFrame when no CSV is found.
    """
    path: Path | None = None
    if season:
        if league == "wnba":
            year = _wnba_season_year(season)
            candidate = _PLAYER_POSITIONS_DIR / f"wnba_{year}_players.csv"
        else:
            end_year = _nba_season_end_year(season)
            candidate = _PLAYER_POSITIONS_DIR / f"nba_{end_year}_players.csv"

        if candidate.exists():
            path = candidate
        else:
            msg = f"  ⚠ season positions file not found: {candidate.name}"
            if league == "nba":
                msg += " — trying player_positions.csv"
            print(msg)

    if path is None and league == "nba":
        fallback = _PLAYER_POSITIONS_DIR / "player_positions.csv"
        if fallback.exists():
            path = fallback

    if path is None:
        print("  ⚠ No player-positions CSV found — pos fallback unavailable")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "pos" not in df.columns:
        print(f"  ⚠ {path.name} missing 'pos' column — skipping")
        return pd.DataFrame()

    # Prefer name_s26 (ASCII / API-aligned) when the season CSV has it.
    name_cols = [c for c in ("name_s26", "name") if c in df.columns]
    if not name_cols:
        print(f"  ⚠ {path.name} missing 'name'/'name_s26' — skipping")
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    pos = df["pos"].astype(str).str.strip()
    valid_pos = pos.ne("") & df["pos"].notna()
    for col in name_cols:
        keys = df.loc[valid_pos, col].map(_norm_player_name)
        part = pd.DataFrame({"_name_lower": keys, "pos_csv": pos[valid_pos]})
        part = part[part["_name_lower"].ne("")]
        parts.append(part)

    if not parts:
        print(f"  ⚠ {path.name} produced no usable name/pos rows — skipping")
        return pd.DataFrame()

    lookup = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset="_name_lower", keep="first"
    )
    print(
        f"  loaded {len(lookup):,} player-position entries from {path.name} "
        f"(season={season or 'fallback'})"
    )
    return lookup


def _to_pos(value: object, *, allowed: frozenset[str]) -> str | None:
    """Canonicalize a raw position label; return it only if in ``allowed``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    canon = _START_POS_CANONICAL.get(raw, raw.upper())
    return canon if canon in allowed else None


def enrich_pos(
    df: pd.DataFrame,
    player_positions: pd.DataFrame | None = None,
    *,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    """Add a ``pos`` column from the season positions CSV, then tracking.

    Priority:
    1. Season player-positions CSV lookup (accent-normalized name).
    2. Tracking ``start_position`` for rows still missing.

    Allowed labels:
    * NBA  — fine slots only (PG / SG / SF / PF / C); coarse G/F ignored.
    * WNBA — Basketball-Reference labels (G / F / C and G-F / F-C hybrids),
      plus fine slots if tracking ever provides them.

    Falls back to ``None`` when neither source has an allowed position.
    """
    out = df.copy()
    out["pos"] = None
    allowed = _WNBA_POS if league == "wnba" else _FINE_POS

    if (
        player_positions is not None
        and not player_positions.empty
        and "player_name" in out.columns
    ):
        pos_map = (
            player_positions.drop_duplicates(subset="_name_lower")
            .assign(
                pos_csv=lambda d: d["pos_csv"].map(
                    lambda v: _to_pos(v, allowed=allowed)
                )
            )
            .dropna(subset=["pos_csv"])
            .set_index("_name_lower")["pos_csv"]
        )
        name_key = out["player_name"].map(_norm_player_name)
        out["pos"] = name_key.map(pos_map)
        matched = int(out["pos"].notna().sum())
        print(f"  pos CSV: matched {matched:,} / {len(out):,} rows")

    if "start_position" in out.columns:
        from_track = out["start_position"].map(
            lambda v: _to_pos(v, allowed=allowed)
        )
        missing_before = int(out["pos"].isna().sum())
        if missing_before:
            out["pos"] = out["pos"].fillna(from_track)
            filled = missing_before - int(out["pos"].isna().sum())
            if filled:
                label = (
                    "G/F/C (+ hybrids)"
                    if league == "wnba"
                    else "PG/SG/SF/PF/C"
                )
                print(
                    f"  pos tracking fallback: filled {filled:,} / {missing_before:,} "
                    f"missing with {label}"
                )

    return out


def _snake_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize API / DB column names to snake_case (``GAME_DATE`` → ``game_date``)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out.columns = [_normalize_col(str(c)) for c in out.columns]
    return out


def _drop_fetched_at(df: pd.DataFrame) -> pd.DataFrame:
    out = _snake_columns(df)
    if out.empty or "fetched_at" not in out.columns:
        return out
    return out.drop(columns=["fetched_at"])


def _prepare_tracking(df: pd.DataFrame) -> pd.DataFrame:
    out = _drop_fetched_at(df)
    rename = {src: dst for src, dst in _TRACKING_SILVER_RENAME.items() if src in out.columns}
    return out.rename(columns=rename)


def _nba_game_id_filter(season_type: str) -> tuple[str, dict]:
    if season_type == "Regular Season":
        return "game_id LIKE %(prefix)s", {"prefix": "002%"}
    if season_type == "Playoffs":
        return "game_id LIKE %(prefix)s", {"prefix": "004%"}
    raise ValueError(f"Unknown season_type: {season_type!r}")


def read_raw_tables(
    season: str,
    season_type: str,
    *,
    league: LeagueKey = "nba",
) -> dict[str, pd.DataFrame]:
    """Load the five raw game-log tables for one season slice from Supabase."""
    cfg = LEAGUES[league]
    tables = cfg.raw_table_by_dataset
    player_base_table = tables["player_base"]

    if league == "nba":
        game_filter, extra = _nba_game_id_filter(season_type)
        where = f"season_year = %(season)s AND {game_filter}"
        params: dict = {"season": season, **extra}
        tracking_where = (
            f"game_id IN ("
            f"SELECT DISTINCT game_id FROM raw.{player_base_table} "
            f"WHERE season_year = %(season)s AND {game_filter}"
            f")"
        )
    else:
        where = "season_year = %(season)s"
        params = {"season": season}
        tracking_where = (
            f"game_id IN ("
            f"SELECT DISTINCT game_id FROM raw.{player_base_table} "
            f"WHERE season_year = %(season)s"
            f")"
        )

    out: dict[str, pd.DataFrame] = {}
    for dataset in _DATASET_KEYS:
        table = tables[dataset]
        table_where = tracking_where if dataset == "start_positions" else where
        df = read_df(table, schema="raw", where=table_where, params=params)
        if dataset == "start_positions":
            out[dataset] = _prepare_tracking(df) if not df.empty else df
        else:
            out[dataset] = _drop_fetched_at(df) if not df.empty else df
        print(f"  read raw.{table} — {len(out[dataset]):,} rows")
    return out


def _redundant_cols(prefix: str) -> list[str]:
    suffixes = [
        "season_year_base",
        "team_abbreviation_base",
        "team_name_base",
        "game_date_base",
        "matchup_base",
        "wl_base",
        "min_base",
        "season_year_adv",
        "team_abbreviation_adv",
        "team_name_adv",
        "game_date_adv",
        "matchup_adv",
        "wl_adv",
        "min_adv",
        "available_flag_base",
        "available_flag_adv",
        # team_base-only path (no team_adv merge → no _base/_adv suffixes)
        "season_year",
        "team_abbreviation",
        "team_name",
        "game_date",
        "matchup",
        "wl",
        "min",
        "available_flag",
    ]
    return [f"{prefix}{s}" for s in suffixes]


def _drop_silver_junk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(
        columns=[c for c in out.columns if "_rank" in c],
        errors="ignore",
    )
    out = out.drop(
        columns=[c for c in out.columns if c.endswith("_adv")],
        errors="ignore",
    )
    out = out.drop(
        columns=[c for c in _redundant_cols("team_") if c in out.columns],
        errors="ignore",
    )
    out = out.drop(
        columns=[c for c in _redundant_cols("opp_") if c in out.columns],
        errors="ignore",
    )
    return out.loc[:, ~out.columns.duplicated()]


def merge_gamelogs(
    player_base: pd.DataFrame,
    player_adv: pd.DataFrame,
    team_base: pd.DataFrame,
    team_adv: pd.DataFrame,
    start_positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge five raw tables into one player-game frame (snake_case)."""
    if player_adv is not None and not player_adv.empty:
        player_merged = player_base.merge(
            player_adv,
            on=["game_id", "player_id", "team_id"],
            suffixes=("", "_adv"),
        )
    else:
        player_merged = player_base.copy()

    if start_positions is not None and not start_positions.empty:
        track_cols = ["game_id", "player_id"] + [
            c
            for c in start_positions.columns
            if c not in ("game_id", "player_id") and c not in player_merged.columns
        ]
        player_merged = player_merged.merge(
            start_positions[track_cols],
            on=["game_id", "player_id"],
            how="left",
        )

    if team_adv is not None and not team_adv.empty:
        team_merged = team_base.merge(
            team_adv,
            on=["game_id", "team_id"],
            suffixes=("_base", "_adv"),
        )
    else:
        team_merged = team_base.copy()

    opp_team = team_merged.copy()
    opp_team = opp_team.add_prefix("team_").rename(
        columns={
            "team_game_id": "game_id",
            "team_team_id": "opp_team_id",
        }
    )
    opp_team.columns = [
        ("opp_" + col[5:]) if col.startswith("team_") else col
        for col in opp_team.columns
    ]

    team_merged = team_merged.add_prefix("team_").rename(
        columns={
            "team_game_id": "game_id",
            "team_team_id": "team_id",
        }
    )

    final = player_merged.merge(team_merged, on=["game_id", "team_id"], how="left")
    final = final.merge(opp_team, on="game_id", how="left")
    final = final.loc[final["team_id"] != final["opp_team_id"]]

    return _drop_silver_junk(final)


def upsert_silver(df: pd.DataFrame, *, league: LeagueKey = "nba") -> None:
    """Upsert a merged frame into ``silver.nba_player_gamelogs`` / ``wnba_player_gamelogs``."""
    if df.empty:
        print("  no rows to upsert")
        return
    table = _SILVER_TABLE[league]
    upsert_df(
        table,
        df,
        schema="silver",
        lineage_col="built_at",
    )


def build_silver(
    season: str,
    season_type: str = "Regular Season",
    *,
    league: Literal["nba", "wnba"] = "nba",
    db_upsert: bool = True,
    raw_frames: dict[str, pd.DataFrame] | None = None,
    auto_scrape_rotowire: bool = False,
) -> pd.DataFrame:
    """Merge raw tables → silver frame; optionally upsert to Supabase.

    Pipeline: merge five endpoints → ``pos`` (tracking + CSV) → Rotowire
    ``game_total`` / ``team_spread`` (NBA) → optional silver upsert.

    Parameters
    ----------
    season:
        e.g. ``\"2025-26\"`` (NBA) or ``\"2025\"`` (WNBA).
    season_type:
        ``\"Regular Season\"`` or ``\"Playoffs\"`` (stamped on every row).
    league:
        ``\"nba\"`` → ``silver.nba_player_gamelogs``;
        ``\"wnba\"`` → ``silver.wnba_player_gamelogs``.
    db_upsert:
        When True, write the merged frame to the silver schema.
    raw_frames:
        Optional in-memory frames keyed by dataset name
        (``player_base``, ``player_adv``, ``team_base``, ``team_adv``,
        ``start_positions``). When omitted, tables are read from Supabase.
    auto_scrape_rotowire:
        When True and the Rotowire CSV is missing, scrape it automatically
        (NBA only).
    """
    if league not in LEAGUES:
        raise ValueError(f"Unknown league: {league!r}")

    label = LEAGUES[league].label
    print(f"── Silver ({label}): {season} {season_type} ──")

    if raw_frames is None:
        print("  loading raw.* from Supabase…")
        frames = read_raw_tables(season, season_type, league=league)
    else:
        frames = {}
        for key in _DATASET_KEYS:
            if key not in raw_frames:
                continue
            df = raw_frames[key]
            if key == "start_positions":
                frames[key] = _prepare_tracking(df) if not df.empty else df
            else:
                frames[key] = _drop_fetched_at(df) if not df.empty else df

    for key in ("player_base", "team_base"):
        if key not in frames or frames[key].empty:
            raise ValueError(f"Missing or empty raw frame: {key}")

    for key in ("player_adv", "team_adv"):
        if key not in frames or frames[key].empty:
            print(f"  ⚠ raw.{key} empty — silver merge will omit advanced stats")

    df = merge_gamelogs(
        frames["player_base"],
        frames.get("player_adv", pd.DataFrame()),
        frames["team_base"],
        frames.get("team_adv", pd.DataFrame()),
        start_positions=frames.get("start_positions"),
    )
    df = df.copy()
    df["season_type"] = season_type

    print(f"  merged — {df.shape}")

    positions = load_player_positions(season, league=league)
    df = enrich_pos(df, positions, league=league)

    rw = load_rotowire(
        season,
        league=league,
        auto_scrape=auto_scrape_rotowire,
    )
    df = enrich_rotowire(df, rw)

    if db_upsert:
        upsert_silver(df, league=league)

    print(f"✓ Silver frame — {len(df):,} rows, {len(df.columns)} columns")
    return df


def fetch_and_build_silver(
    season: str,
    season_type: str = "Regular Season",
    *,
    league: Literal["nba", "wnba"] = "nba",
    db_upsert: bool = True,
    auto_scrape_rotowire: bool = False,
    datasets: list[str] | None = None,
    parallel: bool = True,
    checkpoint_path: str | None = None,
    batch_size: int = 100,
    start_position_delay: float = 0.3,
    start_position_workers: int = 8,
    run_all_batches: bool = True,
) -> pd.DataFrame:
    """Fetch raw endpoints, then merge + enrich into silver in one call.

    Equivalent to ``GameLogs(...).fetch(...)`` followed by
    ``build_silver(..., raw_frames=logs.data)``.
    """
    from src.pipeline.fetch import GameLogs

    logs = GameLogs(season=season, season_type=season_type, league=league)
    logs.fetch(
        datasets=datasets,
        parallel=parallel,
        db_upsert=db_upsert,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        start_position_delay=start_position_delay,
        start_position_workers=start_position_workers,
        run_all_batches=run_all_batches,
    )
    return build_silver(
        season,
        season_type,
        league=league,
        db_upsert=db_upsert,
        raw_frames=logs.data,
        auto_scrape_rotowire=auto_scrape_rotowire,
    )


def _parse_cli_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(
        description="Merge raw.* → silver.* (NBA/WNBA)",
    )
    p.add_argument("--league", choices=("nba", "wnba"), default="nba")
    p.add_argument("--season", default=None)
    p.add_argument(
        "--season-type",
        default="Regular Season",
        help='e.g. "Regular Season" or "Playoffs"',
    )
    p.add_argument("--no-db-upsert", action="store_true")
    p.add_argument(
        "--auto-scrape-rotowire",
        action="store_true",
        help="If Rotowire CSV is missing, scrape it (NBA only)",
    )
    p.add_argument(
        "--fetch",
        action="store_true",
        help="Also fetch raw endpoints before building silver (one-shot)",
    )
    p.add_argument("--sequential", action="store_true", help="With --fetch: sequential API calls")
    p.add_argument("--checkpoint", default=None, help="With --fetch: start_positions checkpoint")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--start-position-delay", type=float, default=0.3)
    p.add_argument("--start-position-workers", type=int, default=8)
    p.add_argument("--one-batch", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m src.pipeline.clean --league nba --season 2025-26``."""
    args = _parse_cli_args(argv)
    season = args.season or LEAGUES[args.league].default_season
    db_upsert = not args.no_db_upsert

    if args.fetch:
        fetch_and_build_silver(
            season,
            args.season_type,
            league=args.league,
            db_upsert=db_upsert,
            auto_scrape_rotowire=args.auto_scrape_rotowire,
            parallel=not args.sequential,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            start_position_delay=args.start_position_delay,
            start_position_workers=args.start_position_workers,
            run_all_batches=not args.one_batch,
        )
    else:
        build_silver(
            season,
            args.season_type,
            league=args.league,
            db_upsert=db_upsert,
            auto_scrape_rotowire=args.auto_scrape_rotowire,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
