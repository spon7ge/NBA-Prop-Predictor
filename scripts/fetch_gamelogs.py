"""
End-to-end NBA game log fetch + position merge + bookmaker name canon + Rotowire lines.

Equivalent to notebooks/fetch_data.ipynb + human_layer merge, with repo-relative paths.

Default inputs (repo-relative): positions ``data/raw/player_positions/nba_2026_players.csv``;
START_POSITION checkpoint ``data/raw/cache/tracking_checkpoint.csv`` (override with ``--checkpoint``).
Game IDs already in the checkpoint are skipped for boxscore fetches only. The four league-wide pulls
(player/team base + advanced) still run every time.

Examples (run from repository root):

  python scripts/fetch_gamelogs.py

  python scripts/fetch_gamelogs.py --skip-nba-fetch \\
      --parquet data/raw/playoff_stats/_last_fetch.parquet

  python scripts/fetch_gamelogs.py --skip-rotowire \\
      --rotowire-csv data/raw/rotowire/rotowire_nba_2025.csv
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_suffix_re = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?$", re.I)


def _checkpoint_summary(path: Path) -> tuple[bool, int, int]:
    """Return (exists, n_rows, n_unique_game_id)."""
    if not path.exists():
        return False, 0, 0
    ck = pd.read_csv(path, dtype=str, low_memory=False)
    n_rows = len(ck)
    if "GAME_ID" not in ck.columns:
        return True, n_rows, 0
    return True, n_rows, ck["GAME_ID"].nunique()


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
        "GAME_DATE":         df["GAME_DATE"],
        "TEAM_ABBREVIATION": df["HOME_TEAM"],
        "TEAM_SPREAD":       home_line,
        "GAME_TOTAL":        total,
    })
    away = pd.DataFrame({
        "GAME_DATE":         df["GAME_DATE"],
        "TEAM_ABBREVIATION": df["AWAY_TEAM"],
        "TEAM_SPREAD":       -home_line,
        "GAME_TOTAL":        total,
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch NBA logs, merge positions & Rotowire lines.")
    p.add_argument("--season", default="2025-26", help="NBA season string for NBAGameLogs")
    p.add_argument(
        "--season-type",
        default="Playoffs",
        help="Regular Season | Playoffs",
    )
    p.add_argument(
        "--positions",
        default="data/raw/player_positions/nba_2026_players.csv",
        help="Player positions CSV (columns name_s26, pos, age)",
    )
    p.add_argument(
        "--reference-season-csv",
        default="data/raw/season_stats/S26.csv",
        help="Reference roster names for bookmaker alias mapping",
    )
    p.add_argument(
        "--out",
        default="data/raw/playoff_stats/P26.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--checkpoint",
        default="data/raw/cache/tracking_checkpoint.csv",
        help="START_POSITION resume checkpoint (relative to repo root unless absolute)",
    )
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--start-position-delay", type=float, default=2.5)
    p.add_argument("--start-position-workers", type=int, default=5)
    p.add_argument(
        "--is-playoff-flag",
        type=int,
        default=1,
        choices=(0, 1),
        help="Value written to IS_PLAYOFF on output",
    )

    p.add_argument(
        "--rotowire-season",
        default="2025",
        help="Rotowire API season filter (same as notebook TARGET_SEASON)",
    )
    p.add_argument(
        "--rotowire-csv",
        default="data/raw/rotowire/rotowire_nba_2025.csv",
        help="Path for Rotowire scrape output / read when --skip-rotowire",
    )
    p.add_argument(
        "--skip-rotowire",
        action="store_true",
        help="Do not run Playwright scraper; read --rotowire-csv",
    )
    p.add_argument(
        "--rotowire-headed",
        action="store_true",
        help="Non-headless browser for Rotowire",
    )

    p.add_argument(
        "--db-upsert",
        action="store_true",
        help=(
            "Upsert each raw DataFrame (player_base, player_adv, team_base, "
            "team_adv, start_positions) into its raw.* Supabase table right "
            "after fetching. Requires SUPABASE_DB_URL in .env and migration "
            "scripts/migrations/001_raw_gamelogs.sql to have been applied."
        ),
    )
    p.add_argument(
        "--skip-nba-fetch",
        action="store_true",
        help="Skip NBA API + start-position fetch; load frame from --parquet",
    )
    p.add_argument(
        "--parquet",
        default="",
        help="With --skip-nba-fetch, load this parquet (written when --save-parquet is set)",
    )
    p.add_argument(
        "--save-parquet",
        default="",
        help="If set, save raw merged NBA frame (before positions) to this path for reuse",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.nbaPlayerLogs import NBAGameLogs
    from src.scrapers.rotowire_scraper import run_scrape as run_rotowire_scrape

    pos_path = PROJECT_ROOT / args.positions
    ref_path = PROJECT_ROOT / args.reference_season_csv
    out_path = PROJECT_ROOT / args.out
    rotowire_path = (
        Path(args.rotowire_csv)
        if Path(args.rotowire_csv).is_absolute()
        else PROJECT_ROOT / args.rotowire_csv
    )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    if args.skip_nba_fetch:
        if not args.parquet:
            print("Error: --skip-nba-fetch requires --parquet", file=sys.stderr)
            return 1
        pq = Path(args.parquet)
        if not pq.is_absolute():
            pq = PROJECT_ROOT / pq
        print(f"Loading NBA frame from {pq}")
        df = pd.read_parquet(pq)
    else:
        print("── NBA: fetch + build ──")
        ck_path = checkpoint.resolve()
        ck_ok, ck_rows, ck_games = _checkpoint_summary(ck_path)
        print(
            f"  START_POSITION checkpoint: {ck_path} "
            f"(exists={ck_ok}, rows={ck_rows}, unique_GAME_ID={ck_games})"
        )
        logs = NBAGameLogs(season=args.season, season_type=args.season_type)
        logs.fetch(
            batch_size=args.batch_size,
            start_position_delay=args.start_position_delay,
            start_position_workers=args.start_position_workers,
            checkpoint_path=str(ck_path),
            db_upsert=args.db_upsert,
        )
        df = logs.build().get_df()
        print(f"✓ Built — shape: {df.shape}")
        if args.save_parquet:
            pq_out = Path(args.save_parquet)
            if not pq_out.is_absolute():
                pq_out = PROJECT_ROOT / pq_out
            pq_out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pq_out, index=False)
            print(f"✓ Saved raw NBA frame → {pq_out}")

    print("── Positions ──")
    if not pos_path.exists():
        print(f"Error: positions file not found: {pos_path}", file=sys.stderr)
        return 1
    pos = pd.read_csv(pos_path).rename(
        columns={"name_s26": "PLAYER_NAME", "pos": "POS", "age": "AGE"}
    )
    df = df.merge(pos, on="PLAYER_NAME", how="left")
    df["IS_PLAYOFF"] = args.is_playoff_flag

    print("── Bookmaker name canon ──")
    if not ref_path.exists():
        print(f"Error: reference CSV not found: {ref_path}", file=sys.stderr)
        return 1
    ref = pd.read_csv(ref_path)
    if "PLAYER_NAME" not in ref.columns:
        print("Error: reference CSV must contain PLAYER_NAME", file=sys.stderr)
        return 1
    canon = _build_name_canon_map(ref["PLAYER_NAME"])
    df = apply_bookmaker_name_aliases(df, canon)

    df["STARTING"] = df["START_POSITION"].notna().astype(int)
    df["PTS_PER_MIN"] = df["PTS"] / df["MIN"].replace(0, np.nan)
    df["AST_PER_MIN"] = df["AST"] / df["MIN"].replace(0, np.nan)
    df["REB_PER_MIN"] = df["REB"] / df["MIN"].replace(0, np.nan)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs", na=False).astype(int)

    pos_fill = df["POS"].fillna("UNK").astype(str)
    le = LabelEncoder()
    df["POSITION_ENCODED"] = le.fit_transform(pos_fill)

    missing = sorted(set(df["PLAYER_NAME"].astype(str)) - set(ref["PLAYER_NAME"].astype(str)))
    print(f"Output unique names: {df['PLAYER_NAME'].nunique()} | missing vs reference: {len(missing)}")
    if missing:
        print("Missing (sample):", missing[:50])

    print("── Rotowire ──")
    if args.skip_rotowire:
        if not rotowire_path.exists():
            print(f"Error: Rotowire CSV not found: {rotowire_path}", file=sys.stderr)
            return 1
        ro = pd.read_csv(rotowire_path)
    else:
        asyncio.run(
            run_rotowire_scrape(
                season=args.rotowire_season,
                output_file=rotowire_path,
                headless=not args.rotowire_headed,
            )
        )
        ro = pd.read_csv(rotowire_path)

    ro_long = _build_rotowire_long(ro)
    df = merge_rotowire(df, ro_long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)} rows → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
