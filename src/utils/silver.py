"""Silver layer: merge raw game-log tables, clean names, join reference data.

Reads from ``raw.*`` (Supabase) or in-memory bronze frames, produces one flat
player-game DataFrame compatible with the legacy ``S26.csv`` / ``P26.csv`` shape
(through rotowire + positions + name canon). Model features (``PTS_PER_MIN``,
``POSITION_ENCODED``, …) belong in gold — not here.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.db import read_df
from src.utils.nbaPlayerLogs import _TRACK_V3_LEGACY_ALIASES

RAW_TABLES: tuple[str, ...] = (
    "player_base",
    "player_adv",
    "team_base",
    "team_adv",
    "start_positions",
)

_suffix_re = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?$", re.I)

# Invert tracking aliases: POSITION → START_POSITION, SPEED → SPD, …
_TRACKING_SILVER_RENAME = {
    source: legacy for legacy, source in _TRACK_V3_LEGACY_ALIASES.items()
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_POSITIONS = _PROJECT_ROOT / "data/raw/player_positions/player_positions.csv"
_DEFAULT_REFERENCE = _PROJECT_ROOT / "data/raw/season_stats/S26.csv"
_DEFAULT_ROTOWIRE = _PROJECT_ROOT / "data/raw/rotowire/rotowire_nba_2025.csv"


def _resolve_data_path(path: str | Path) -> Path:
    """Resolve repo-relative data paths regardless of notebook/script cwd."""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (_PROJECT_ROOT / p).resolve()


def _db_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """``snake_case`` Postgres columns → ``SCREAMING_SNAKE`` modeling columns."""
    out = df.copy()
    out.columns = [c.upper() for c in out.columns]
    drop = [c for c in ("FETCHED_AT",) if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out


def _prepare_start_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = _db_to_frame(df)
    rename = {src: dst for src, dst in _TRACKING_SILVER_RENAME.items() if src in out.columns}
    return out.rename(columns=rename)


def _season_type_game_id_filter(season_type: str) -> tuple[str, dict]:
    if season_type == "Regular Season":
        return "game_id LIKE %(prefix)s", {"prefix": "002%"}
    if season_type == "Playoffs":
        return "game_id LIKE %(prefix)s", {"prefix": "004%"}
    raise ValueError(f"Unknown season_type: {season_type!r}")


def read_raw_tables(season: str, season_type: str) -> dict[str, pd.DataFrame]:
    """Load bronze game-log tables for one season slice from Supabase."""
    game_filter, extra = _season_type_game_id_filter(season_type)
    where = f"season_year = %(season)s AND {game_filter}"
    params = {"season": season, **extra}
    # start_positions has game_id/player_id only — scope via player_base game IDs.
    start_positions_where = (
        "game_id IN ("
        "SELECT DISTINCT game_id FROM raw.player_base "
        f"WHERE season_year = %(season)s AND {game_filter}"
        ")"
    )

    out: dict[str, pd.DataFrame] = {}
    for table in RAW_TABLES:
        table_where = start_positions_where if table == "start_positions" else where
        df = read_df(table, where=table_where, params=params)
        if table == "start_positions":
            out[table] = _prepare_start_positions(df) if not df.empty else df
        else:
            out[table] = _db_to_frame(df) if not df.empty else df
        print(f"  read raw.{table} — {len(out[table]):,} rows")
    return out


def merge_gamelogs(
    player_base: pd.DataFrame,
    player_adv: pd.DataFrame,
    team_base: pd.DataFrame,
    team_adv: pd.DataFrame,
    start_positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge five bronze tables into one player-game frame (no name/rotowire joins)."""
    if player_adv is not None and not player_adv.empty:
        player_merged = player_base.merge(
            player_adv,
            on=["GAME_ID", "PLAYER_ID", "TEAM_ID"],
            suffixes=("", "_adv"),
        )
    else:
        player_merged = player_base.copy()

    if start_positions is not None and not start_positions.empty:
        track_cols = ["GAME_ID", "PLAYER_ID"] + [
            c
            for c in start_positions.columns
            if c not in ("GAME_ID", "PLAYER_ID") and c not in player_merged.columns
        ]
        player_merged = player_merged.merge(
            start_positions[track_cols],
            on=["GAME_ID", "PLAYER_ID"],
            how="left",
        )

    if team_adv is not None and not team_adv.empty:
        team_merged = team_base.merge(
            team_adv,
            on=["GAME_ID", "TEAM_ID"],
            suffixes=("_base", "_adv"),
        )
    else:
        team_merged = team_base.copy()

    opp_team = team_merged.copy()
    opp_team = opp_team.add_prefix("TEAM_").rename(columns={
        "TEAM_GAME_ID": "GAME_ID",
        "TEAM_TEAM_ID": "OPP_TEAM_ID",
    })
    opp_team.columns = [
        ("OPP_" + col[5:]) if col.startswith("TEAM_") else col
        for col in opp_team.columns
    ]

    team_merged = team_merged.add_prefix("TEAM_").rename(columns={
        "TEAM_GAME_ID": "GAME_ID",
        "TEAM_TEAM_ID": "TEAM_ID",
    })

    final = player_merged.merge(team_merged, on=["GAME_ID", "TEAM_ID"], how="left")
    final = final.merge(opp_team, on="GAME_ID", how="left")
    final = final.loc[final["TEAM_ID"] != final["OPP_TEAM_ID"]]

    return _drop_silver_junk(final)


def _redundant_cols(prefix: str) -> list[str]:
    suffixes = [
        "SEASON_YEAR_base", "TEAM_ABBREVIATION_base", "TEAM_NAME_base",
        "GAME_DATE_base", "MATCHUP_base", "WL_base", "MIN_base",
        "SEASON_YEAR_adv", "TEAM_ABBREVIATION_adv", "TEAM_NAME_adv",
        "GAME_DATE_adv", "MATCHUP_adv", "WL_adv", "MIN_adv",
        "AVAILABLE_FLAG_base", "AVAILABLE_FLAG_adv",
        # team_base-only path (no team_adv merge → no _base/_adv suffixes)
        "SEASON_YEAR", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_DATE", "MATCHUP", "WL", "MIN", "AVAILABLE_FLAG",
    ]
    return [f"{prefix}{s}" for s in suffixes]


def _drop_silver_junk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in out.columns if "_RANK" in c], errors="ignore")
    out = out.drop(columns=[c for c in out.columns if c.endswith("_adv")], errors="ignore")
    out = out.drop(columns=[c for c in _redundant_cols("TEAM_") if c in out.columns], errors="ignore")
    out = out.drop(columns=[c for c in _redundant_cols("OPP_") if c in out.columns], errors="ignore")
    return out.loc[:, ~out.columns.duplicated()]


# ── Name canon (from fetch_gamelogs.py) ───────────────────────────────────────

def _norm_player_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.replace("'", "")
    name = re.sub(r"[^A-Za-z0-9 ]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    name = _suffix_re.sub("", name).strip()
    return re.sub(r"\s+", " ", name).strip()


def _build_name_canon_map(reference_names: pd.Series) -> dict[str, set[str]]:
    m: dict[str, set[str]] = {}
    for _n in reference_names.dropna().astype(str).unique():
        m.setdefault(_norm_player_name(_n), set()).add(_n)
    return m


def apply_bookmaker_name_aliases(df: pd.DataFrame, canon_map: dict[str, set[str]]) -> pd.DataFrame:
    out = df.copy()
    out["PLAYER_NAME"] = [
        sorted(canon_map.get(_norm_player_name(n), {str(n)}))[0]
        for n in out["PLAYER_NAME"].astype(str)
    ]
    return out


# ── Rotowire (from fetch_gamelogs.py) ─────────────────────────────────────────

def _build_rotowire_long(rotowire_df: pd.DataFrame) -> pd.DataFrame:
    df = rotowire_df.copy()

    df[["AWAY_TEAM", "HOME_TEAM"]] = (
        df["Game"].astype(str).str.split(r"\s*@\s*", n=1, expand=True)
    )
    df["AWAY_TEAM"] = df["AWAY_TEAM"].str.strip().str.upper()
    df["HOME_TEAM"] = df["HOME_TEAM"].str.strip().str.upper()

    month_str = df["Tipoff"].astype(str).str.split().str[0]
    is_first_half = month_str.isin(["Oct", "Nov", "Dec"])
    season_start = pd.to_numeric(df["Season"], errors="coerce")
    inferred_year = np.where(is_first_half, season_start, season_start + 1)

    parsed = pd.to_datetime(
        df["Tipoff"].astype(str) + " " + inferred_year.astype(str),
        format="%b %d %I:%M %p %Y",
        errors="coerce",
    )
    df["GAME_DATE"] = parsed.dt.normalize()

    score = df["Score"].astype(str).str.split("-", n=1, expand=True)
    df["AWAY_SCORE"] = pd.to_numeric(score[0], errors="coerce")
    df["HOME_SCORE"] = pd.to_numeric(score[1], errors="coerce")

    home_line = pd.to_numeric(df["Home_Line"], errors="coerce")
    total = pd.to_numeric(df["Over_Under"], errors="coerce")

    home = pd.DataFrame({
        "GAME_DATE": df["GAME_DATE"],
        "TEAM_ABBREVIATION": df["HOME_TEAM"],
        "TEAM_SPREAD": home_line,
        "GAME_TOTAL": total,
    })
    away = pd.DataFrame({
        "GAME_DATE": df["GAME_DATE"],
        "TEAM_ABBREVIATION": df["AWAY_TEAM"],
        "TEAM_SPREAD": -home_line,
        "GAME_TOTAL": total,
    })
    return pd.concat([home, away], ignore_index=True).dropna(
        subset=["GAME_DATE", "TEAM_ABBREVIATION"]
    )


def merge_rotowire(player_df: pd.DataFrame, rotowire_long: pd.DataFrame) -> pd.DataFrame:
    out = player_df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.normalize()
    for col in ("TEAM_SPREAD", "GAME_TOTAL"):
        if col in out.columns:
            out = out.drop(columns=[col])
    out = out.merge(rotowire_long, on=["GAME_DATE", "TEAM_ABBREVIATION"], how="left")

    if "TEAM_SPREAD_ODDS" in out.columns:
        needs_fill = out["TEAM_SPREAD"].isna() | (out["TEAM_SPREAD"] == 0)
        out.loc[needs_fill, "TEAM_SPREAD"] = out.loc[needs_fill, "TEAM_SPREAD_ODDS"]
        out["TEAM_SPREAD"] = out["TEAM_SPREAD"] + 0.0
    if "GAME_TOTAL_ODDS" in out.columns:
        out["GAME_TOTAL"] = out["GAME_TOTAL"].fillna(out["GAME_TOTAL_ODDS"])

    return out


def merge_positions(df: pd.DataFrame, positions_path: str | Path) -> pd.DataFrame:
    pos = pd.read_csv(positions_path)
    rename = {"pos": "POS", "age": "AGE"}
    if "name_s26" in pos.columns:
        rename["name_s26"] = "PLAYER_NAME"
    elif "name" in pos.columns:
        rename["name"] = "PLAYER_NAME"
    else:
        raise ValueError("Positions CSV must contain 'name_s26' or 'name' column")
    pos = pos.rename(columns=rename)[["PLAYER_NAME", "POS", "AGE"]]
    return df.merge(pos, on="PLAYER_NAME", how="left")


def enrich_gamelogs_silver(
    df: pd.DataFrame,
    *,
    positions_path: str | Path,
    reference_season_csv: str | Path,
    rotowire_csv: str | Path,
    is_playoff: int = 0,
    skip_rotowire: bool = False,
) -> pd.DataFrame:
    """Apply positions, name canon, and rotowire to an already-merged frame."""
    out = merge_positions(df, positions_path)
    out["IS_PLAYOFF"] = is_playoff

    ref_path = Path(reference_season_csv)
    ref = pd.read_csv(ref_path, low_memory=False)
    if "PLAYER_NAME" not in ref.columns:
        raise ValueError("Reference CSV must contain PLAYER_NAME")
    canon = _build_name_canon_map(ref["PLAYER_NAME"])
    out = apply_bookmaker_name_aliases(out, canon)

    if not skip_rotowire:
        ro_path = Path(rotowire_csv)
        ro_long = _build_rotowire_long(pd.read_csv(ro_path))
        out = merge_rotowire(out, ro_long)

    return out


# ── Main entry ────────────────────────────────────────────────────────────────

def build_gamelogs_silver(
    season: str,
    season_type: str,
    *,
    raw_frames: dict[str, pd.DataFrame] | None = None,
    positions_path: str | Path = _DEFAULT_POSITIONS,
    reference_season_csv: str | Path = _DEFAULT_REFERENCE,
    rotowire_csv: str | Path = _DEFAULT_ROTOWIRE,
    is_playoff: int | None = None,
    skip_rotowire: bool = False,
) -> pd.DataFrame:
    """Merge raw tables + positions + name canon + rotowire → silver frame."""
    if is_playoff is None:
        is_playoff = 1 if season_type == "Playoffs" else 0

    print(f"── Silver: {season} {season_type} ──")

    if raw_frames is None:
        print("  loading raw.* from Supabase…")
        raw_frames = read_raw_tables(season, season_type)
    else:
        raw_frames = {
            k: (_prepare_start_positions(v) if k == "start_positions" else _db_to_frame(v))
            for k, v in raw_frames.items()
        }

    for key in ("player_base", "team_base"):
        if key not in raw_frames or raw_frames[key].empty:
            raise ValueError(f"Missing or empty raw frame: {key}")

    for key in ("player_adv", "team_adv"):
        if key not in raw_frames or raw_frames[key].empty:
            print(f"  ⚠ raw.{key} empty — silver merge will omit advanced stats")

    positions_path = _resolve_data_path(positions_path)
    if not positions_path.exists():
        raise FileNotFoundError(f"Positions file not found: {positions_path}")
    ref_path = _resolve_data_path(reference_season_csv)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {ref_path}")
    rotowire_path = _resolve_data_path(rotowire_csv)
    if not skip_rotowire and not rotowire_path.exists():
        raise FileNotFoundError(f"Rotowire CSV not found: {rotowire_path}")

    df = merge_gamelogs(
        raw_frames["player_base"],
        raw_frames["player_adv"],
        raw_frames["team_base"],
        raw_frames["team_adv"],
        start_positions=raw_frames.get("start_positions"),
    )
    print(f"  merged — {df.shape}")

    df = enrich_gamelogs_silver(
        df,
        positions_path=positions_path,
        reference_season_csv=ref_path,
        rotowire_csv=rotowire_path,
        is_playoff=is_playoff,
        skip_rotowire=skip_rotowire,
    )

    missing = sorted(
        set(df["PLAYER_NAME"].astype(str))
        - set(pd.read_csv(ref_path, low_memory=False)["PLAYER_NAME"].astype(str))
    )
    print(f"  names — {df['PLAYER_NAME'].nunique()} unique | missing vs reference: {len(missing)}")
    if missing:
        print(f"  missing sample: {missing[:20]}")

    print(f"✓ Silver frame — {len(df):,} rows, {len(df.columns)} columns")
    return df
