"""
Prediction ledger and results reconciliation.

Ledger files (append-only, newline-delimited JSON):
    data/logs/predictions.jsonl  — one row per prop × bookmaker × run
    data/logs/slates.jsonl       — one row per leg in every greedy slate (long format)
    data/logs/results.jsonl      — one row per reconciled prop after gamelog update

Dedup key for predictions : (DATE, PLAYER_NAME, MARKET, LINE_BOOKMAKER, LINE)
Dedup key for slates      : (DATE, SLATE_ID, PLAYER_NAME, MARKET, LINE_BOOKMAKER, LINE)

Re-runs at the *same line* replace the prior entry for that key.
Re-runs after a *line move* keep both entries (line history is preserved).
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

LOGS_DIR          = Path("data/logs")
PREDICTIONS_FILE  = LOGS_DIR / "predictions.jsonl"
SLATES_FILE       = LOGS_DIR / "slates.jsonl"
RESULTS_FILE      = LOGS_DIR / "results.jsonl"

# Columns used to identify a unique prediction entry
_PRED_KEY   = ["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER", "LINE"]
_SLATE_KEY  = ["DATE", "SLATE_ID", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER", "LINE"]
_RESULT_KEY = ["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER"]

# Mapping from model MARKET label → base_df column
_STAT_COL = {
    "PTS": "PTS",
    "AST": "AST",
    "REB": "REB",
    "TOV": "TOV",
    "FTA": "FTA",
    "3PM": "FG3M",
    "BLK": "BLK",
    "STL": "STL",
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(_json_safe(r)) for r in df.to_dict("records")) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row)) + "\n")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if np.isnan(x) else x
    if isinstance(obj, (str, bool, int)) or obj is None:
        return obj
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    return obj


def _dedup_and_write(new_df: pd.DataFrame, path: Path, key_cols: list[str]) -> None:
    """
    Drop any existing rows whose key_cols match a row in new_df, then append new_df.
    Re-reads the file so the full history is preserved except for exact key matches.
    """
    existing = _read_jsonl(path)
    if not existing.empty and all(c in existing.columns for c in key_cols):
        new_keys = set(zip(*[new_df[c] for c in key_cols]))
        mask = existing.apply(
            lambda r: tuple(r[c] for c in key_cols) in new_keys, axis=1
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
        _write_jsonl(combined, path)
    else:
        _append_jsonl(new_df.to_dict("records"), path)


# ── Core API ──────────────────────────────────────────────────────────────────

def snapshot(
    all_line_probs: pd.DataFrame,
    slate_paths: dict,
    date: str | None = None,
    run_timestamp: str | None = None,
) -> None:
    """
    Log today's predictions and slates to the ledger files.

    Args:
        all_line_probs:  Enriched DataFrame from main() (all bookmakers combined).
                         Must contain LINE_BOOKMAKER column.
        slate_paths:     {bookmaker: {'2leg': path_str, '3leg': path_str}}
                         Paths to the already-written slate JSON files.
        date:            Override date string 'YYYY-MM-DD' (defaults to today).
        run_timestamp:   Override ISO timestamp (defaults to now).
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")
    if run_timestamp is None:
        run_timestamp = datetime.now().isoformat(timespec="seconds")

    # ── Predictions ──────────────────────────────────────────────────────────
    preds = all_line_probs.copy()
    preds["DATE"]          = date
    preds["RUN_TIMESTAMP"] = run_timestamp
    preds["SIDE"]          = preds.apply(_derive_side, axis=1)

    _dedup_and_write(preds, PREDICTIONS_FILE, _PRED_KEY)
    print(f"[log] predictions  {len(preds):>4} rows  ({date}  ts={run_timestamp})")

    # ── Slates ───────────────────────────────────────────────────────────────
    slate_rows = []
    for bookmaker, type_paths in slate_paths.items():
        for slate_type, path_str in type_paths.items():
            path = Path(path_str)
            if not path.exists():
                continue
            try:
                pairs = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                continue
            if not pairs:
                continue

            n_legs = 3 if "3" in slate_type else 2
            for idx, pair in enumerate(pairs):
                slate_id = f"{date}_{bookmaker.replace(' ', '_')}_{slate_type}_{idx:04d}"
                for leg in range(1, n_legs + 1):
                    slate_rows.append({
                        "DATE":          date,
                        "RUN_TIMESTAMP": run_timestamp,
                        "SLATE_ID":      slate_id,
                        "SLATE_TYPE":    slate_type,
                        "LINE_BOOKMAKER":bookmaker,
                        "PARLAY_PROB":   pair.get("PARLAY_PROB"),
                        "EV":            pair.get("EV"),
                        "KELLY":         pair.get("KELLY"),
                        "PLAYER_NAME":   pair.get(f"NAME {leg}"),
                        "MARKET":        pair.get(f"MARKET {leg}"),
                        "LINE":          pair.get(f"LINE {leg}"),
                        "SIDE":          pair.get(f"SIDE {leg}"),
                        "PREDICTION":    pair.get(f"PREDICTION {leg}"),
                        "MODEL_PROB":    pair.get(f"MODEL_PROB {leg}"),
                        "TEAM":          pair.get(f"TEAM {leg}"),
                        "OPPONENT":      pair.get(f"OPPONENT {leg}"),
                        "SPREAD":        pair.get(f"SPREAD {leg}"),
                        "TOTAL":         pair.get(f"TOTAL {leg}"),
                    })

    if slate_rows:
        slate_df = pd.DataFrame(slate_rows)
        _dedup_and_write(slate_df, SLATES_FILE, _SLATE_KEY)
        print(f"[log] slates       {len(slate_rows):>4} leg rows  ({date})")
    else:
        print("[log] slates       no data to log")


def reconcile(date: str, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match logged predictions for `date` against actual stats in base_df.

    Scores only the most-recent prediction per (DATE, PLAYER_NAME, MARKET, LINE_BOOKMAKER)
    so line moves don't inflate the row count.

    Appends new results to results.jsonl, skipping any already reconciled.
    Returns the reconciled DataFrame for the given date.
    """
    preds = _read_jsonl(PREDICTIONS_FILE)
    if preds.empty:
        print("[log] No predictions on file")
        return pd.DataFrame()

    day_preds = preds[preds["DATE"] == date].copy()
    if day_preds.empty:
        print(f"[log] No predictions found for {date}")
        return pd.DataFrame()

    # Keep only the latest prediction per key (handles line moves)
    day_preds["RUN_TIMESTAMP"] = pd.to_datetime(day_preds["RUN_TIMESTAMP"])
    latest = (
        day_preds.sort_values("RUN_TIMESTAMP")
        .groupby(["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER"])
        .last()
        .reset_index()
    )

    # Skip already-reconciled rows
    existing = _read_jsonl(RESULTS_FILE)
    if not existing.empty and all(c in existing.columns for c in _RESULT_KEY):
        done = set(zip(*[existing[c] for c in _RESULT_KEY]))
        latest = latest[
            ~latest.apply(lambda r: tuple(r[c] for c in _RESULT_KEY) in done, axis=1)
        ]

    if latest.empty:
        print(f"[log] {date} already fully reconciled")
        return existing[existing["DATE"] == date] if not existing.empty else pd.DataFrame()

    # Look up actual stats
    game_day = base_df[base_df["GAME_DATE"] == date].copy()
    result_rows = []
    for _, pred in latest.iterrows():
        actual_stat, actual_min, hit, miss_reason = _score_prediction(pred, game_day)
        result_rows.append({
            **pred.to_dict(),
            "ACTUAL_STAT":    actual_stat,
            "ACTUAL_MIN":     actual_min,
            "HIT":            hit,
            "MISS_REASON":    miss_reason,
            "RECONCILED_AT":  datetime.now().isoformat(timespec="seconds"),
        })

    result_df = pd.DataFrame(result_rows)
    _append_jsonl(result_rows, RESULTS_FILE)

    hits  = int(result_df["HIT"].sum())
    total = len(result_df)
    pct   = hits / total if total else 0
    print(f"[log] reconciled   {total:>4} props for {date}  →  {hits}/{total} hit ({pct:.1%})")
    return result_df


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _derive_side(row) -> str:
    q50  = float(row.get("STAT_Q50", 0))
    line = float(row.get("LINE", 0))
    if q50 > line:
        return "over"
    if q50 < line:
        return "under"
    po = float(row.get("P_OVER", 0.5))
    pu = float(row.get("P_UNDER", 0.5))
    return "over" if po >= pu else "under"


def _score_prediction(pred: pd.Series, game_day: pd.DataFrame) -> tuple:
    """
    Returns (actual_stat, actual_min, hit, miss_reason).
    miss_reason is one of:
        dnp            — player not in gamelog (injury / rest)
        blowout        — actual < STAT_Q10 (minutes collapsed)
        sharp_line     — market IMP_PROB was 45–55% (coin flip)
        model_miss_low — predicted over, finished under line but above Q10
        model_miss_high— predicted under, finished over line
        clean_hit      — hit with margin > 10% of line
        squeaker       — hit with margin ≤ 10% of line
    """
    name   = pred["PLAYER_NAME"]
    market = str(pred.get("MARKET", "PTS"))
    line   = float(pred.get("LINE", 0))
    side   = str(pred.get("SIDE", "over"))

    player_game = game_day[game_day["PLAYER_NAME"] == name]

    if player_game.empty:
        return None, None, False, "dnp"

    stat_col = _STAT_COL.get(market, market)
    row = player_game.iloc[0]

    actual_stat = float(row[stat_col]) if stat_col in row.index else None
    actual_min  = float(row["MIN"])    if "MIN"     in row.index else None

    if actual_stat is None:
        return None, actual_min, False, "dnp"

    hit = (actual_stat > line) if side == "over" else (actual_stat < line)

    if not hit:
        q10      = float(pred.get("STAT_Q10", line))
        imp_over = float(pred.get("IMP_PROB_OVER", 0.5))
        if actual_stat < q10:
            reason = "blowout"
        elif 0.45 <= imp_over <= 0.55:
            reason = "sharp_line"
        elif side == "over":
            reason = "model_miss_low"
        else:
            reason = "model_miss_high"
    else:
        margin = abs(actual_stat - line)
        reason = "clean_hit" if line == 0 or margin / line > 0.10 else "squeaker"

    return actual_stat, actual_min, hit, reason


# ── Analysis helpers ──────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    """Load the full predictions ledger."""
    return _read_jsonl(PREDICTIONS_FILE)


def load_slates() -> pd.DataFrame:
    """Load the full slates ledger (long format — one row per leg)."""
    return _read_jsonl(SLATES_FILE)


def load_results() -> pd.DataFrame:
    """Load the full results ledger."""
    return _read_jsonl(RESULTS_FILE)


def hit_rate_by(group_col: str) -> pd.DataFrame:
    """
    Aggregate hit rate grouped by any column.
    Common values: 'MARKET', 'LINE_BOOKMAKER', 'MISS_REASON', 'DATE'
    """
    results = load_results()
    if results.empty or group_col not in results.columns:
        return pd.DataFrame()
    return (
        results.groupby(group_col)["HIT"]
        .agg(hits="sum", total="count", hit_rate="mean")
        .round(3)
        .reset_index()
        .sort_values("hit_rate", ascending=False)
    )


def miss_breakdown() -> pd.DataFrame:
    """Count of each MISS_REASON label across all reconciled props."""
    results = load_results()
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby("MISS_REASON")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def calibration(bucket_size: float = 0.05) -> pd.DataFrame:
    """
    Model calibration: predicted P_OVER bucketed vs actual hit rate.
    A well-calibrated model should show actual_rate ≈ P_BUCKET.
    Only evaluates 'over' side predictions.
    """
    results = load_results()
    if results.empty:
        return pd.DataFrame()
    over = results[results["SIDE"] == "over"].copy()
    over["P_OVER"] = pd.to_numeric(over["P_OVER"], errors="coerce")
    over["P_BUCKET"] = (over["P_OVER"] // bucket_size * bucket_size).round(3)
    return (
        over.groupby("P_BUCKET")["HIT"]
        .agg(hits="sum", total="count", actual_rate="mean")
        .round(3)
        .reset_index()
    )


def slate_results() -> pd.DataFrame:
    """
    Join slates to results on (DATE, PLAYER_NAME, MARKET, LINE_BOOKMAKER).
    Each row is a slate leg with HIT and ACTUAL_STAT filled in.
    Group by SLATE_ID to check full parlay wins:

        sr = slate_results()
        parlay_wins = sr.groupby('SLATE_ID')['HIT'].all()
    """
    slates  = load_slates()
    results = load_results()
    if slates.empty or results.empty:
        return pd.DataFrame()

    key_cols    = ["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER"]
    result_cols = key_cols + ["HIT", "ACTUAL_STAT", "ACTUAL_MIN", "MISS_REASON"]

    return slates.merge(
        results[result_cols],
        on=key_cols,
        how="left",
    )


def line_movement(date: str | None = None) -> pd.DataFrame:
    """
    Show all players whose line changed during the day (multiple RUN_TIMESTAMP entries
    for the same player/market/bookmaker on the same date).
    Pass a specific date, or None to scan all dates.
    """
    preds = load_predictions()
    if preds.empty:
        return pd.DataFrame()
    if date:
        preds = preds[preds["DATE"] == date]

    counts = preds.groupby(["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER"])["LINE"].nunique()
    movers = counts[counts > 1].reset_index().rename(columns={"LINE": "N_LINES"})

    detail = preds.merge(
        movers[["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER"]],
        on=["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER"],
    ).sort_values(["PLAYER_NAME", "MARKET", "LINE_BOOKMAKER", "RUN_TIMESTAMP"])

    return detail[["DATE", "RUN_TIMESTAMP", "PLAYER_NAME", "MARKET",
                   "LINE_BOOKMAKER", "LINE", "STAT_Q50", "P_OVER", "SIDE"]]

# To reconcile the predictions, run the following:
# from src.utils import log
# log.reconcile("2026-05-02", base_df)  