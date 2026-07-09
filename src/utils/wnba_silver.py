"""WNBA silver layer: merge ``raw.wnba_*`` tables into one flat player-game frame.

No Rotowire, positions CSV, or name canon — just merge + drop rank/redundant cols.
Schema: ``db/migrations/011_wnba_silver_gamelogs.sql``.
"""

from __future__ import annotations

import pandas as pd

from src.utils.db import read_df
from src.utils.silver import (
    _db_to_frame,
    _prepare_start_positions,
    merge_gamelogs,
)
from src.utils.playerLogs import RAW_TABLE_BY_DATASET

# In-memory / read_raw keys → Supabase ``raw.wnba_*`` table names.
RAW_KEYS: tuple[str, ...] = tuple(RAW_TABLE_BY_DATASET.keys())


def _season_game_id_prefix(season_type: str) -> str:
    if season_type == "Regular Season":
        return "102%"
    if season_type == "Playoffs":
        return "104%"
    raise ValueError(f"Unknown season_type: {season_type!r}")


def _season_filters(season: str, season_type: str) -> dict:
    return {
        "eq": {"season_year": season},
        "like": {"game_id": _season_game_id_prefix(season_type)},
    }


def read_wnba_raw_tables(season: str, season_type: str) -> dict[str, pd.DataFrame]:
    """Load WNBA bronze tables for one season slice from Supabase."""
    filters = _season_filters(season, season_type)
    game_filter, extra = (
        "game_id LIKE %(prefix)s",
        {"prefix": _season_game_id_prefix(season_type)},
    )
    where = f"season_year = %(season)s AND {game_filter}"
    params = {"season": season, **extra}

    out: dict[str, pd.DataFrame] = {}
    for key, table in RAW_TABLE_BY_DATASET.items():
        if key == "start_positions":
            if "player_base" not in out:
                pb = read_df(
                    RAW_TABLE_BY_DATASET["player_base"],
                    where=where,
                    params=params,
                    **filters,
                )
                out["player_base"] = _db_to_frame(pb) if not pb.empty else pb
            game_ids = (
                out["player_base"]["GAME_ID"].astype(str).unique().tolist()
                if not out["player_base"].empty
                else []
            )
            df = (
                read_df(table, in_={"game_id": game_ids})
                if game_ids
                else pd.DataFrame()
            )
        else:
            df = read_df(table, where=where, params=params, **filters)
        if key == "start_positions":
            out[key] = _prepare_start_positions(df) if not df.empty else df
        else:
            out[key] = _db_to_frame(df) if not df.empty else df
        print(f"  read raw.{table} — {len(out[key]):,} rows")
    return out


def build_wnba_gamelogs_silver(
    season: str,
    season_type: str,
    *,
    raw_frames: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Merge WNBA raw tables → flat silver frame (no external enrichments)."""
    print(f"── WNBA silver: {season} {season_type} ──")

    if raw_frames is None:
        print("  loading raw.wnba_* from Supabase…")
        raw_frames = read_wnba_raw_tables(season, season_type)
    else:
        raw_frames = {
            k: (_prepare_start_positions(v) if k == "start_positions" else _db_to_frame(v))
            for k, v in raw_frames.items()
        }

    for key in ("player_base", "team_base"):
        if key not in raw_frames or raw_frames[key].empty:
            raise ValueError(f"Missing or empty raw frame: {key}")

    for key in ("player_adv", "team_adv", "start_positions"):
        if key not in raw_frames or raw_frames[key].empty:
            print(f"  ⚠ raw.{RAW_TABLE_BY_DATASET[key]} empty — merge will omit those stats")

    df = merge_gamelogs(
        raw_frames["player_base"],
        raw_frames.get("player_adv"),
        raw_frames["team_base"],
        raw_frames.get("team_adv"),
        start_positions=raw_frames.get("start_positions"),
    )
    df["SEASON_TYPE"] = season_type
    print(f"✓ WNBA silver frame — {len(df):,} rows, {len(df.columns)} columns")
    return df
