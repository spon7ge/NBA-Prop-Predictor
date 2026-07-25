"""
Run the live prop prediction pipeline and write results to ml.live_prop_predictions.

Workflow
--------
1. Load def-rating context once via nba_api (load_opp_def_ratings).
2. Load quantile model bundles (min + ppm/apm/rpm) for the given league.
3. For each prop market (player_points, player_assists, player_rebounds):
   a. Pull the latest DFS lines from raw Supabase tables.
   b. Call predict_rate() to get per-player quantile predictions + game context.
   c. For each bookmaker in those lines, call line_probs_for_market() to get
      P_OVER / P_UNDER for that book's specific line.
   d. Enrich each row with:
      - vs_opp: historical stats vs tonight's opponent (n_games, avg, over-rate).
      - form:   hit-rate vs the book line over the last 5 / 10 / 15 games.
4. Upsert everything into ml.live_prop_predictions.

Examples (run from repository root)
-------------------------------------
    python scripts/run_live_props.py --league nba
    python scripts/run_live_props.py --league wnba --date 2026-07-14
    python scripts/run_live_props.py --league nba --season-type "Regular Season" --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError).
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pipeline.predict import (
    _load_silver,
    line_probs_for_market,
    load_latest_odds,
    load_opp_def_ratings,
    load_quantile_bundles,
    predict_rate,
)
from src.utils.db import read_df, upsert_live_prop_predictions
from src.utils.distributions import run_count_simulation, run_pts_simulation

# ── constants ─────────────────────────────────────────────────────────────────

_PROPS = ["player_points", "player_assists", "player_rebounds"]

_MARKET_LABEL = {
    "player_points":   "PTS",
    "player_assists":  "AST",
    "player_rebounds": "REB",
}

_STAT_COL = {
    "player_points":   "pts",
    "player_assists":  "ast",
    "player_rebounds": "reb",
}

_SIM_FN = {
    "player_points":   run_pts_simulation,
    "player_assists":  run_count_simulation,
    "player_rebounds": run_count_simulation,
}


_RATE_BUNDLE_KEY = {
    "player_points":   "ppm",
    "player_assists":  "apm",
    "player_rebounds": "rpm",
}


# ── enrichment helpers ────────────────────────────────────────────────────────

def _opp_from_matchup(matchup: object, team_abbr: object) -> str | None:
    """Parse opponent tricode from silver ``matchup`` (e.g. ``PHX vs. LAS`` / ``PHX @ GSV``)."""
    if matchup is None or (isinstance(matchup, float) and pd.isna(matchup)):
        return None
    m = str(matchup).strip().upper()
    team = str(team_abbr).strip().upper() if team_abbr is not None else ""
    if " VS." in m or " VS " in m:
        parts = m.replace(" VS.", " VS ").split(" VS ")
        if len(parts) != 2:
            return None
        a, b = parts[0].strip(), parts[1].strip()
        if team and a == team:
            return b
        if team and b == team:
            return a
        return b  # default: right side is visitor/opponent label
    if " @ " in m:
        parts = m.split(" @ ")
        if len(parts) != 2:
            return None
        a, b = parts[0].strip(), parts[1].strip()
        # "PHX @ GSV" → team PHX, opp GSV
        if team and a == team:
            return b
        if team and b == team:
            return a
        return b
    return None


def _vs_opp_stats(
    silver: pd.DataFrame,
    player_name: str,
    opp_abbr: str | None,
    stat_col: str,
    line: float | None,
) -> dict:
    """Compute vs-opponent history for one player × stat."""
    empty = {"vs_opp_n_games": None, "vs_opp_avg_stat": None, "vs_opp_over_rate": None}
    if opp_abbr is None or (isinstance(opp_abbr, float) and pd.isna(opp_abbr)) or not opp_abbr:
        return empty

    pdf = silver[silver["player_name"] == player_name].copy()
    if pdf.empty or stat_col not in pdf.columns:
        return empty

    want = str(opp_abbr).strip().upper()

    # Prefer dedicated opp abbr column when populated
    opp_col = next(
        (
            c
            for c in (
                "opp_team_abbreviation",
                "opp_abbreviation",
                "opp_abbr",
            )
            if c in pdf.columns and pdf[c].notna().any()
        ),
        None,
    )
    if opp_col is not None:
        opp_series = pdf[opp_col].astype(str).str.strip().str.upper()
        vs = pdf[opp_series == want]
    elif "matchup" in pdf.columns:
        team_col = "team_abbreviation" if "team_abbreviation" in pdf.columns else None
        parsed = [
            _opp_from_matchup(
                row["matchup"],
                row[team_col] if team_col else None,
            )
            for _, row in pdf.iterrows()
        ]
        pdf = pdf.assign(_parsed_opp=parsed)
        vs = pdf[pdf["_parsed_opp"].astype(str).str.upper() == want]
    else:
        return empty

    if vs.empty:
        return empty

    n = len(vs)
    avg = float(vs[stat_col].mean())
    over_rate = (
        float((vs[stat_col] > line).mean()) if line is not None else None
    )
    return {"vs_opp_n_games": n, "vs_opp_avg_stat": round(avg, 2), "vs_opp_over_rate": over_rate}


def _def_rating_ranks(def_ratings: dict) -> dict[int, int]:
    """TEAM_ID → rank (1 = lowest / stingiest DEF_RATING)."""
    if not def_ratings:
        return {}
    ordered = sorted(
        (
            (tid, float(info["DEF_RATING"]))
            for tid, info in def_ratings.items()
            if info and info.get("DEF_RATING") is not None
        ),
        key=lambda x: x[1],
    )
    return {tid: rank for rank, (tid, _) in enumerate(ordered, start=1)}


def _load_silver_history(league: str, seasons_back: int = 4) -> pd.DataFrame:
    """Multi-season silver for vs-opp history (current-season slice is too thin)."""
    table = f"{league}_player_gamelogs"
    today = date.today()
    if league == "wnba":
        min_season = str(today.year - seasons_back)
    else:
        start = today.year if today.month >= 10 else today.year - 1
        min_start = start - seasons_back
        min_season = f"{min_start}-{str(min_start + 1)[-2:]}"

    df = read_df(
        table,
        schema="silver",
        where="season_year >= %(min_season)s",
        params={"min_season": min_season},
    )
    print(
        f"  silver.{table} history: {len(df):,} rows "
        f"(season_year >= {min_season})"
    )
    return df


def _pred_context_by_player(preds: pd.DataFrame) -> dict[str, dict]:
    """Index predict_rate context columns by player (lost in line_probs_for_market)."""
    out: dict[str, dict] = {}
    if preds.empty:
        return out
    for _, r in preds.iterrows():
        name = str(r.get("PLAYER_NAME", "")).strip()
        if not name:
            continue
        out[name] = {
            "PLAYER_TEAM":  r.get("PLAYER_TEAM"),
            "OPP_TEAM":     r.get("OPP_TEAM"),
            "HOME":         r.get("HOME"),
            "OPP_DEF_RATING": r.get("OPP_DEF_RATING"),
            "OPP_PACE":     r.get("OPP_PACE"),
        }
    return out
    """Index predict_rate context columns by player (lost in line_probs_for_market)."""
    out: dict[str, dict] = {}
    if preds.empty:
        return out
    for _, r in preds.iterrows():
        name = str(r.get("PLAYER_NAME", "")).strip()
        if not name:
            continue
        out[name] = {
            "PLAYER_TEAM":  r.get("PLAYER_TEAM"),
            "OPP_TEAM":     r.get("OPP_TEAM"),
            "HOME":         r.get("HOME"),
            "OPP_DEF_RATING": r.get("OPP_DEF_RATING"),
            "OPP_PACE":     r.get("OPP_PACE"),
        }
    return out


def _form_stats(
    silver: pd.DataFrame,
    player_name: str,
    stat_col: str,
    line: float | None,
) -> dict:
    """Compute hit-rate vs the book line for the last 5/10/15 games."""
    empty = {"over_l5": None, "over_l10": None, "over_l15": None}
    if line is None:
        return empty

    pdf = silver[silver["player_name"] == player_name].sort_values("game_date")
    if pdf.empty or stat_col not in pdf.columns:
        return empty

    vals = pdf[stat_col].dropna()

    def _rate(n):
        tail = vals.tail(n)
        return round(float((tail > line).mean()), 3) if len(tail) >= 1 else None

    return {"over_l5": _rate(5), "over_l10": _rate(10), "over_l15": _rate(15)}


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    league: str,
    game_date: str,
    season_type: str,
    models_dir: Path,
    *,
    dry_run: bool = False,
) -> pd.DataFrame:
    run_at = datetime.now(timezone.utc)

    # ── 1. load def-rating context (hits nba_api once) ──────────────────────
    print(f"\nFetching defensive ratings ({season_type})…")
    try:
        def_ratings, league_avg_def_rtg, league_avg_pace = load_opp_def_ratings(
            season_type=season_type,
            league=league,
        )
    except Exception as exc:
        print(f"  WARNING: could not load def ratings ({exc}); context will be NaN")
        def_ratings, league_avg_def_rtg, league_avg_pace = {}, None, None

    # ── 2. load quantile models ──────────────────────────────────────────────
    bundles = load_quantile_bundles(league, models_dir=models_dir)

    # ── 3. pre-load silver (predict_rate loads current season; history for vs-opp)
    print("\nPre-loading silver gamelogs for enrichment…")
    silver = _load_silver(league)
    silver_hist = _load_silver_history(league)

    all_frames: list[pd.DataFrame] = []

    for prop in _PROPS:
        market    = _MARKET_LABEL[prop]
        stat_col  = _STAT_COL[prop]
        sim_fn    = _SIM_FN[prop]
        rate_key  = _RATE_BUNDLE_KEY[prop]

        print(f"\n--- {market} ({prop}) ---")

        # ── 3a. load DFS lines ──────────────────────────────────────────────
        all_lines = load_latest_odds(league=league, region="dfs", prop=prop)
        if all_lines.empty:
            print(f"  No DFS lines found for {prop}; skipping.")
            continue

        names = all_lines["NAME"].dropna().unique().tolist()
        print(f"  {len(names)} unique players with DFS lines")

        # ── 3b. model inference + game context (once per prop) ───────────────
        preds = predict_rate(
            names,
            game_date,
            prop,
            league=league,
            min_bundle=bundles["min"],
            rate_bundle=bundles[rate_key],
            def_ratings=def_ratings,
            league_avg_def_rtg=league_avg_def_rtg,
            league_avg_pace=league_avg_pace,
        )
        if preds.empty:
            print(f"  No predictions generated for {prop}; skipping.")
            continue
        print(f"  {len(preds)} players predicted")

        # Context lives on preds; line_probs_for_market drops those columns.
        pred_ctx = _pred_context_by_player(preds)
        def_ranks = _def_rating_ranks(def_ratings)
        abbr_to_id: dict[str, int] = {}
        if {"team_abbreviation", "team_id"}.issubset(silver.columns):
            for _, sr in (
                silver.dropna(subset=["team_abbreviation", "team_id"])
                .drop_duplicates("team_abbreviation")
                .iterrows()
            ):
                abbr_to_id[str(sr["team_abbreviation"]).strip().upper()] = int(sr["team_id"])

        # ── 3c. probability scoring per bookmaker ────────────────────────────
        for book, book_lines in all_lines.groupby("BOOKMAKER"):
            results = line_probs_for_market(preds, book_lines, sim_fn)
            if results.empty:
                continue
            results["BOOKMAKER"] = book

            # ── 3d. form + vs-opp enrichment ────────────────────────────────
            enriched_rows: list[dict] = []
            for _, row in results.iterrows():
                name = str(row["PLAYER_NAME"]).strip()
                ctx  = pred_ctx.get(name, {})
                opp  = ctx.get("OPP_TEAM")
                line = row.get("LINE")
                # line may come from results (STAT column) vs book_lines
                if pd.isna(line) if isinstance(line, float) else line is None:
                    # try to get it from book_lines directly
                    matched = book_lines[
                        book_lines["NAME"].str.strip() == name
                    ]
                    line = float(matched["LINE"].iloc[0]) if not matched.empty else None

                vs_opp = _vs_opp_stats(silver_hist, name, opp, stat_col, line)
                form   = _form_stats(silver, name, stat_col, line)

                opp_rank = None
                if opp is not None and not (isinstance(opp, float) and pd.isna(opp)):
                    tid = abbr_to_id.get(str(opp).strip().upper())
                    if tid is not None:
                        opp_rank = def_ranks.get(tid)

                enriched_rows.append({
                    **row.to_dict(),
                    "PLAYER_TEAM":    ctx.get("PLAYER_TEAM"),
                    "OPP_TEAM":       opp,
                    "HOME":           ctx.get("HOME"),
                    "OPP_DEF_RATING": ctx.get("OPP_DEF_RATING"),
                    "OPP_PACE":       ctx.get("OPP_PACE"),
                    "opp_def_rating_rank": opp_rank,
                    **vs_opp,
                    **form,
                    "run_at":    run_at,
                    "league":    league,
                    "game_date": game_date,
                })

            all_frames.append(pd.DataFrame(enriched_rows))

    if not all_frames:
        print("\nNo data to write.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # ── rename to snake_case columns that match the DB table ─────────────────
    _rename = {
        "PLAYER_NAME":           "player_name",
        "PLAYER_TEAM":           "team_abbr",
        "OPP_TEAM":              "opponent_abbr",
        "HOME":                  "is_home",
        "MARKET":                "market",
        "BOOKMAKER":             "bookmaker",
        "LINE":                  "line",
        "STAT_Q10":              "stat_q10",
        "STAT_Q50":              "stat_q50",
        "STAT_Q90":              "stat_q90",
        "MIN_Q10":               "min_q10",
        "MIN_Q50":               "min_q50",
        "MIN_Q90":               "min_q90",
        "P_OVER":                "p_over",
        "P_UNDER":               "p_under",
        "OPP_DEF_RATING":        "opp_def_rating",
        "OPP_PACE":              "opp_pace",
        "LEAGUE_AVG_DEF_RATING": "league_avg_def_rating",
        "LEAGUE_AVG_PACE":       "league_avg_pace",
    }
    combined = combined.rename(columns=_rename)

    # findOpp returns 0/1; DB column is boolean
    if "is_home" in combined.columns:
        combined["is_home"] = combined["is_home"].map(
            lambda v: None if v is None or (isinstance(v, float) and pd.isna(v))
            else bool(int(v))
        )

    # drop the league column — it's now encoded in the table name
    combined = combined.drop(columns=["league"], errors="ignore")

    print(f"\nTotal enriched rows: {len(combined):,}")
    if dry_run:
        print("\n[dry-run] Sample output:")
        display_cols = [
            c for c in [
                "player_name", "bookmaker", "market", "line",
                "stat_q50", "p_over", "p_under",
                "opponent_abbr", "opp_def_rating",
                "vs_opp_n_games", "over_l10",
            ] if c in combined.columns
        ]
        print(combined[display_cols].head(10).to_string(index=False))
        return combined

    upsert_live_prop_predictions(combined, league=league)
    print(f"\nDone — results written to ml.{league}_live_prop_predictions.")
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate live prop predictions and write to ml.live_prop_predictions."
    )
    p.add_argument(
        "--league",
        choices=("nba", "wnba"),
        default="nba",
        help="League to run (default: nba)",
    )
    p.add_argument(
        "--date",
        default=str(date.today()),
        help="Game date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--season-type",
        default="Regular Season",
        help='NBA season type, e.g. "Regular Season" | "Playoffs" (default: Regular Season)',
    )
    p.add_argument(
        "--models-dir",
        default=str(PROJECT_ROOT / "models" / "saved_models"),
        help="Directory containing .joblib model bundles",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline but do not write to the database",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_pipeline(
        league=args.league,
        game_date=args.date,
        season_type=args.season_type,
        models_dir=Path(args.models_dir),
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
