"""
Prediction ledger and results reconciliation.

Primary store (append semantics with key-based replacement):
    data/logs/predictions.jsonl — one JSON object per line; full history with deduping

Human-readable mirror (same rows as the JSONL ledger, indented envelope):
    data/logs/predictions_ledger.json — ``{ generated_at, n_picks, picks, … }``
    shaped like ``underdog_enriched_*.json``. Rewritten whenever ``snapshot`` updates
    ``predictions.jsonl``.

Further ledgers:

    data/logs/dfs_enriched.jsonl      — one row per pick from dfs_enriched_*.json
    data/logs/dfs_sharp_aligned.jsonl — one row per pick from dfs_sharp_aligned_*.json
    data/logs/slates.jsonl       — one row per exported slate with all legs attached

    data/logs/results.jsonl      — compact reconciled rows (keys + model medians, side, outcomes)

``snapshot`` accepts any ``slate_paths`` shape ``{bookmaker: {"2leg": path, "3leg": path}}``.
Slate JSON may include optional parlay-level fields (e.g. strategy tier); those are
kept on the slate row alongside flattened leg columns and the raw ``LEGS`` list.

Dedup/replace semantics for predictions (JSONL + bundle mirror):

    Dedup key: ``(DATE, PLAYER_NAME, MARKET, LINE_BOOKMAKER, LINE)``

    ``_dedup_and_write``: load existing JSONL, drop every old row whose key appears in the
    *incoming batch*, concatenate ``[remaining_old, incoming_new]``, rewrite the entire file.

    Effects:
      • Re-running ``snapshot`` the same slate day / player / stat / book / numeric line —
        replaces the prior row with the new props (fresh ``RUN_TIMESTAMP``, EV, quantiles…).
      • If the sportsbook moves the numeric ``LINE``, the tuple changes → **both rows stay**
        as separate history rows (line history retained).

Dedup key for slates:
    ``(DATE, SLATE_ID)``
"""


import json
import re
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path

from src.utils.helpers import normalize_game_date_series

LOGS_DIR               = Path("data/logs")
PREDICTIONS_FILE       = LOGS_DIR / "predictions.jsonl"
PREDICTIONS_LEDGER_JSON = LOGS_DIR / "predictions_ledger.json"
DFS_ENRICHED_FILE      = LOGS_DIR / "dfs_enriched.jsonl"
DFS_SHARP_ALIGNED_FILE = LOGS_DIR / "dfs_sharp_aligned.jsonl"
SLATES_FILE            = LOGS_DIR / "slates.jsonl"
RESULTS_FILE      = LOGS_DIR / "results.jsonl"
DFS_ENRICHED_DIR       = Path("data/props/enriched")
SLATES_DIR             = Path("data/props/ev_analysis")

# Columns used to identify a unique prediction entry
_PRED_KEY   = ["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER", "LINE"]
_DFS_PICK_KEY = ["DATE", "PLAYER_NAME", "MARKET", "LINE_BOOKMAKER", "LINE"]
_SLATE_KEY  = ["DATE", "SLATE_ID"]
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

def _rows_for_calendar_date(
    base_df: pd.DataFrame,
    date_str: str,
    *,
    date_col: str = "GAME_DATE",
) -> pd.DataFrame:
    """
    Rows whose calendar day matches ``date_str`` (``YYYY-MM-DD``).

    Uses ``normalize_game_date_series`` so season rows (``YYYY-MM-DD``) and playoff
    rows (ISO timestamps) parse correctly after ``pd.concat`` — plain
    ``pd.to_datetime`` can yield NaT for playoff-only formats (pandas 2.x).
    """
    if date_col not in base_df.columns:
        return pd.DataFrame()
    cal = normalize_game_date_series(base_df[date_col]).dt.strftime("%Y-%m-%d")
    return base_df.loc[cal == date_str].copy()


def _player_game_rows(game_day: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Match PLAYER_NAME with stripped whitespace (prediction vs gamelog spellings)."""
    if game_day.empty or "PLAYER_NAME" not in game_day.columns:
        return pd.DataFrame()
    want = str(player_name).strip()
    mask = game_day["PLAYER_NAME"].astype(str).str.strip() == want
    return game_day.loc[mask]


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
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, np.datetime64):
        t = pd.Timestamp(obj)
        return None if pd.isna(t) else t.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, np.bool_):
        return bool(obj)
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
    Replace-on-key, then rewrite the ledger file.

    1. Read existing rows from ``path`` (if any).
    2. Build the set of key tuples coming from ``new_df`` (often one snapshot batch).
    3. Drop any *existing* row whose key lies in that set.
    4. Concatenate ``[kept_existing, new_df]`` and rewrite ``path``.

    Rows whose key only appears in the incoming batch behave as inserts; overlapping keys
    behave as replacements (fresh run replaces the stale row).

    Same logic is used for slates with ``_SLATE_KEY`` (see module docstring).
    """
    existing = _read_jsonl(path)
    if not existing.empty and all(c in existing.columns for c in key_cols):
        new_keys = set(zip(*[new_df[c] for c in key_cols]))
        mask = existing.apply(
            lambda r: tuple(r[c] for c in key_cols) in new_keys, axis=1
        )
        existing = existing[~mask]
        combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
        _write_jsonl(combined, path)
    else:
        _append_jsonl(new_df.to_dict("records"), path)

    if path.resolve() == PREDICTIONS_FILE.resolve():
        _sync_predictions_ledger_json()


def _sync_predictions_ledger_json() -> None:
    """Rewrite ``predictions_ledger.json`` — envelope + picks, analogous to Underdog enriched JSON."""
    df = _read_jsonl(PREDICTIONS_FILE)
    picks: list = [_json_safe(r) for r in df.to_dict("records")] if not df.empty else []
    payload: dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_picks": len(picks),
        "ledger_source": str(PREDICTIONS_FILE.name),
        "dedupe_key_columns": list(_PRED_KEY),
        "picks": picks,
    }
    if not df.empty and "DATE" in df.columns:
        payload["date_span"] = {
            "min": str(df["DATE"].min()),
            "max": str(df["DATE"].max()),
        }
    if not df.empty and "RUN_TIMESTAMP" in df.columns:
        ts = pd.to_datetime(df["RUN_TIMESTAMP"], errors="coerce").max()
        if pd.notna(ts):
            payload["latest_run_timestamp"] = ts.isoformat()

    PREDICTIONS_LEDGER_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_LEDGER_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _date_from_meta_or_path(meta: dict, path: Path, default: str | None = None) -> str:
    if meta.get("current_date"):
        return str(meta["current_date"])
    match = re.search(r"(\d{8})", path.stem)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return default or datetime.today().strftime("%Y-%m-%d")


def _load_json_payload(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def _dfs_pick_rows_from_file(
    path: Path,
    *,
    run_timestamp: str | None = None,
) -> list[dict]:
    data = _load_json_payload(path)
    if data is None:
        return []
    picks = data.get("picks", []) if isinstance(data, dict) else data
    meta = {k: v for k, v in data.items() if k != "picks"} if isinstance(data, dict) else {}
    if not isinstance(picks, list):
        return []

    date_str = _date_from_meta_or_path(meta, path)
    ts = run_timestamp or meta.get("generated_at") or datetime.now().isoformat(timespec="seconds")
    rows: list[dict] = []
    for pick in picks:
        if not isinstance(pick, dict):
            continue
        model = pick.get("model") or {}
        sharp = pick.get("sharp") or {}
        consensus = pick.get("consensus") or {}
        game_context = pick.get("game_context") or {}
        side = pick.get("side") or model.get("lean")
        rows.append({
            "DATE": date_str,
            "RUN_TIMESTAMP": ts,
            "SOURCE_FILE": path.name,
            "SOURCE_GENERATED_AT": meta.get("generated_at"),
            "LINE_BOOKMAKER": pick.get("platform"),
            "PLAYER_NAME": pick.get("player") or pick.get("display_name"),
            "MARKET": pick.get("market"),
            "LINE": pick.get("dfs_line"),
            "SIDE": str(side).lower() if side is not None else None,
            "P_WIN": pick.get("p_win"),
            "P_OVER": model.get("p_over"),
            "P_UNDER": model.get("p_under"),
            "MIN_Q10": model.get("min_q10"),
            "MIN_Q50": model.get("min_q50"),
            "MIN_Q90": model.get("min_q90"),
            "STAT_Q10": model.get("stat_q10"),
            "STAT_Q50": model.get("stat_q50"),
            "STAT_Q90": model.get("stat_q90"),
            "SHARP_SOURCE": sharp.get("source"),
            "SHARP_LINE": sharp.get("line"),
            "SHARP_LEAN": sharp.get("lean"),
            "CONSENSUS_N_BOOKS_SAME_LINE": consensus.get("n_books_same_line"),
            "GAME_TOTAL": game_context.get("game_total"),
            "SPREAD": game_context.get("spread"),
            "TEAM": pick.get("team_abbr") or pick.get("team"),
            "OPPONENT": pick.get("opponent_abbr") or pick.get("opponent"),
            "TIER": pick.get("tier"),
            "SHARP_ALIGNED": pick.get("sharp_aligned"),
            "PICK": pick,
        })
    return rows


def _track_dfs_pick_files(
    pattern: str,
    ledger_path: Path,
    *,
    enriched_dir: str | Path = DFS_ENRICHED_DIR,
    run_timestamp: str | None = None,
) -> int:
    rows: list[dict] = []
    for path in sorted(Path(enriched_dir).expanduser().glob(pattern)):
        rows.extend(_dfs_pick_rows_from_file(path, run_timestamp=run_timestamp))

    if not rows:
        print(f"[log] {ledger_path.name:<18} no data to log")
        return 0

    df = pd.DataFrame(rows)
    _dedup_and_write(df, ledger_path, _DFS_PICK_KEY)
    print(f"[log] {ledger_path.name:<18} {len(rows):>4} rows")
    return len(rows)


_SLATE_LEG_SUFFIXES = ("_6leg", "_5leg", "_3leg", "_2leg")


def _bookmaker_from_slate_path(path: Path, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    stem = path.stem
    for suffix in _SLATE_LEG_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return {
        "betr": "Betr DFS",
        "draftKings": "DraftKings Pick6",
        "prizepicks": "PrizePicks",
        "underdog": "Underdog",
    }.get(stem, stem)


def _slate_type_from_path(path: Path, fallback: str | None = None) -> str:
    if fallback:
        return str(fallback)
    for suffix in _SLATE_LEG_SUFFIXES:
        if path.stem.endswith(suffix):
            return suffix[1:]
    return "2leg"


def _slate_rows_from_file(
    path: Path,
    *,
    bookmaker: str | None = None,
    slate_type: str | None = None,
    date: str | None = None,
    run_timestamp: str | None = None,
) -> list[dict]:
    pairs = _load_json_payload(path)
    if not isinstance(pairs, list) or not pairs:
        return []

    date_str = date or datetime.today().strftime("%Y-%m-%d")
    ts = run_timestamp or datetime.now().isoformat(timespec="seconds")
    book = _bookmaker_from_slate_path(path, bookmaker)
    slate_kind = _slate_type_from_path(path, slate_type)
    rows: list[dict] = []

    for idx, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            continue
        slate_id = f"{date_str}_{book.replace(' ', '_')}_{slate_kind}_{idx:04d}"
        slate_row = {
            "DATE": date_str,
            "RUN_TIMESTAMP": ts,
            "SLATE_ID": slate_id,
            "SLATE_TYPE": slate_kind,
            "LINE_BOOKMAKER": book,
            "PARLAY_PROB": pair.get("PARLAY_PROB"),
            "EV": pair.get("EV"),
            "EV_DOLLARS": pair.get("EV_DOLLARS"),
            "KELLY": pair.get("KELLY"),
            "PARLAY_N_LEGS": pair.get("N_LEGS"),
            "STRATEGY_TIER": pair.get("STRATEGY_TIER"),
            "COMBO_PROFILE": pair.get("COMBO_PROFILE"),
            "ANCHOR_NAME": pair.get("ANCHOR_NAME"),
            "ANCHOR_WIN_PROB": pair.get("ANCHOR_WIN_PROB"),
            "STAKE_DOLLARS": pair.get("STAKE_DOLLARS"),
            "COMBINED_PAYOUT_MULT": pair.get("COMBINED_PAYOUT_MULT") or pair.get("NET_PAYOUT_MULT"),
            "SLATE_SOURCE": pair.get("SOURCE") or path.name,
        }
        legs: list[dict] = []

        nested_legs = pair.get("LEGS")
        if isinstance(nested_legs, list):
            n_legs = pair.get("N_LEGS") or len(nested_legs)
            slate_row["PARLAY_N_LEGS"] = n_legs
            for leg in nested_legs:
                if not isinstance(leg, dict):
                    continue
                model = leg.get("model") or {}
                game_context = leg.get("game_context") or {}
                side = leg.get("side") or model.get("lean")
                gt_leg = game_context.get("game_total")
                legs.append({
                    "PLAYER_NAME": leg.get("player") or leg.get("display_name"),
                    "MARKET": leg.get("market"),
                    "LINE": leg.get("dfs_line"),
                    "SIDE": str(side).lower() if side is not None else None,
                    "PREDICTION": model.get("stat_q50"),
                    "MODEL_PROB": leg.get("p_win"),
                    "TEAM": leg.get("team_abbr") or leg.get("team"),
                    "OPPONENT": leg.get("opponent_abbr") or leg.get("opponent"),
                    "SPREAD": game_context.get("spread"),
                    "TOTAL": gt_leg,
                    "GAME_TOTAL": gt_leg,
                    "PAYOUT_MULT": pair.get("NET_PAYOUT_MULT"),
                    "OPP_DEF_RATING_RANK": game_context.get("opp_def_rating_rank"),
                    "OPP_PACE_RANK": game_context.get("opp_pace_rank"),
                })
        else:
            n_raw = pair.get("N_LEGS")
            if n_raw is not None:
                try:
                    n_legs = int(n_raw)
                except (TypeError, ValueError):
                    n_legs = 0
            else:
                n_legs = 3 if pair.get("NAME 3") is not None else 2
            if n_legs not in (2, 3, 5, 6):
                n_legs = int(slate_kind.replace("leg", "")) if slate_kind.replace("leg", "").isdigit() else 2
            slate_row["PARLAY_N_LEGS"] = n_legs

            for leg_idx in range(1, n_legs + 1):
                gt_leg = pair.get(f"GAME_TOTAL {leg_idx}")
                tot_leg = pair.get(f"TOTAL {leg_idx}")
                legs.append({
                    "PLAYER_NAME": pair.get(f"NAME {leg_idx}"),
                    "MARKET": pair.get(f"MARKET {leg_idx}"),
                    "LINE": pair.get(f"LINE {leg_idx}"),
                    "SIDE": pair.get(f"SIDE {leg_idx}"),
                    "PREDICTION": pair.get(f"PREDICTION {leg_idx}"),
                    "MODEL_PROB": pair.get(f"MODEL_PROB {leg_idx}"),
                    "TEAM": pair.get(f"TEAM {leg_idx}"),
                    "OPPONENT": pair.get(f"OPPONENT {leg_idx}"),
                    "SPREAD": pair.get(f"SPREAD {leg_idx}"),
                    "TOTAL": tot_leg,
                    "GAME_TOTAL": gt_leg if gt_leg is not None else tot_leg,
                    "PAYOUT_MULT": pair.get(f"PAYOUT_MULT {leg_idx}"),
                    "OPP_DEF_RATING_RANK": pair.get(f"OPP_DEF_RATING_RANK {leg_idx}"),
                    "OPP_PACE_RANK": pair.get(f"OPP_PACE_RANK {leg_idx}"),
                })

        if not legs:
            continue
        slate_row["PLAYER_NAMES"] = " | ".join(
            str(leg["PLAYER_NAME"]) for leg in legs if leg.get("PLAYER_NAME") is not None
        )
        slate_row["LEGS"] = legs
        for leg_idx, leg in enumerate(legs, start=1):
            for key, value in leg.items():
                slate_row[f"{key}_{leg_idx}"] = value
        rows.append(slate_row)
    return rows


# ── Core API ──────────────────────────────────────────────────────────────────
def preds_from_underdog_enriched(
    raw_probs: pd.DataFrame,
    enriched_path: str | Path,
) -> pd.DataFrame:
    """
    Join simulation rows ``raw_probs`` to ``underdog_enriched_*.json`` picks on
    ``(PLAYER_NAME, MARKET, LINE)`` for ledger-compatible flat predictions.

    Picks without a matching ``raw_probs`` row are omitted.
    """
    path = Path(enriched_path).expanduser()
    if raw_probs.empty or not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    picks: list = data.get("picks") or []
    if not picks:
        return pd.DataFrame()
    rp = raw_probs.copy()
    if "PLAYER_NAME" not in rp.columns or "MARKET" not in rp.columns or "LINE" not in rp.columns:
        return pd.DataFrame()
    rp["LINE"] = pd.to_numeric(rp["LINE"], errors="coerce")
    rows: list[dict] = []
    for p in picks:
        name = p.get("player")
        mkt = p.get("market")
        line_raw = p.get("ud_line")
        if name is None or mkt is None or line_raw is None:
            continue
        try:
            line_f = float(line_raw)
        except (TypeError, ValueError):
            continue
        hit = rp[
            (rp["PLAYER_NAME"] == name)
            & (rp["MARKET"] == mkt)
            & np.isclose(rp["LINE"], line_f)
        ]
        if hit.empty:
            continue
        d = hit.iloc[0].to_dict()
        d["LINE_BOOKMAKER"] = "Underdog"
        d["LINE"] = line_f
        rows.append(d)
    return pd.DataFrame(rows)


def snapshot(
    all_line_probs: pd.DataFrame,
    slate_paths: dict,
    date: str | None = None,
    run_timestamp: str | None = None,
) -> None:
    """
    Log today's predictions and slates to the ledger files.

    Args:
        all_line_probs:  Enriched prop-level DataFrame. Must contain ``LINE_BOOKMAKER``.
        slate_paths:     ``{bookmaker: {"2leg": path, "3leg": path, "5leg": path, "6leg": path}}``.
                         Empty or whitespace paths are skipped.
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
    preds["SIDE"] = preds.apply(_derive_side, axis=1)
    adj_cols = [c for c in preds.columns if str(c).startswith("ADJ_")]
    if adj_cols:
        preds = preds.drop(columns=adj_cols)

    _dedup_and_write(preds, PREDICTIONS_FILE, _PRED_KEY)
    print(f"[log] predictions  {len(preds):>4} rows  ({date}  ts={run_timestamp})")

    # ── Slates ───────────────────────────────────────────────────────────────
    slate_rows = []
    for bookmaker, type_paths in slate_paths.items():
        for slate_type, path_str in type_paths.items():
            if not path_str or not str(path_str).strip():
                continue
            path = Path(path_str)
            if not path.exists():
                continue
            slate_rows.extend(
                _slate_rows_from_file(
                    path,
                    bookmaker=bookmaker,
                    slate_type=slate_type,
                    date=date,
                    run_timestamp=run_timestamp,
                )
            )

    if slate_rows:
        slate_df = pd.DataFrame(slate_rows)
        _dedup_and_write(slate_df, SLATES_FILE, _SLATE_KEY)
        print(f"[log] slates       {len(slate_rows):>4} slate rows  ({date})")
    else:
        print("[log] slates       no data to log")


def reconcile(date: str, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match logged predictions for `date` against actual stats in base_df.

    ``base_df`` must contain rows for that **calendar** game day under ``GAME_DATE``
    (season + playoff CSVs if needed). Date matching is normalized so string/datetime
    columns both work.

    Scores only the most-recent prediction per (DATE, PLAYER_NAME, MARKET, LINE_BOOKMAKER)
    so line moves don't inflate the row count.

    Appends compact rows to ``results.jsonl`` (see ``_compact_reconcile_record``),
    skipping any already reconciled. Returns that compact DataFrame for the given date.
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

    # Look up actual stats (calendar-safe GAME_DATE match; see _rows_for_calendar_date)
    game_day = _rows_for_calendar_date(base_df, date)
    if game_day.empty:
        ts = normalize_game_date_series(base_df["GAME_DATE"])
        lo = ts.min()
        hi = ts.max()
        print(
            f"[log] reconcile: no rows in base_df for calendar day {date!r}. "
            f"GAME_DATE range present: {lo} … {hi}. "
            f"Include playoff gamelogs if that slate is playoffs (e.g. concat P26 with S26), "
            f"or run fetch_data so CSVs cover this date."
        )

    result_rows = []
    for _, pred in latest.iterrows():
        actual_stat, actual_min, hit, miss_reason = _score_prediction(pred, game_day)
        actual_game_pts = _game_combined_pts(game_day, pred["PLAYER_NAME"])
        full = {
            **pred.to_dict(),
            "ACTUAL_STAT":    actual_stat,
            "ACTUAL_MIN":     actual_min,
            "HIT":            bool(hit),
            "MISS_REASON":    miss_reason,
            "ACTUAL_GAME_PTS": actual_game_pts,
            "RECON_CONTEXT":  _recon_context_summary(pred, actual_game_pts),
            "RECONCILED_AT":  datetime.now().isoformat(timespec="seconds"),
        }
        result_rows.append(_compact_reconcile_record(full))

    result_df = pd.DataFrame(result_rows)
    _append_jsonl(result_rows, RESULTS_FILE)

    hits  = int(result_df["HIT"].sum())
    total = len(result_df)
    pct   = hits / total if total else 0
    print(f"[log] reconciled   {total:>4} props for {date}  →  {hits}/{total} hit ({pct:.1%})")
    return result_df


def _recon_context_summary(
    _pred: pd.Series,
    actual_game_pts: float | None = None,
) -> str:
    """Optional actual game total for results review."""
    parts: list[str] = []
    if actual_game_pts is not None:
        parts.append(f"actual_pts={actual_game_pts:.0f}")
    return "|".join(parts)


def _game_combined_pts(game_day: pd.DataFrame, player_name: str) -> float | None:
    sub = _player_game_rows(game_day, player_name)
    if sub.empty:
        return None
    r = sub.iloc[0]
    if "TEAM_PTS" not in r.index or "OPP_PTS" not in r.index:
        return None
    tp, opp = r["TEAM_PTS"], r["OPP_PTS"]
    try:
        if pd.isna(tp) or pd.isna(opp):
            return None
        return float(tp) + float(opp)
    except (TypeError, ValueError):
        return None


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


def _compact_reconcile_record(full: dict) -> dict:
    """
    Minimal row shape appended to ``results.jsonl`` (keeps dedupe / slate-join keys).

    ``SIDE`` is model-derived from ``STAT_Q50`` vs ``LINE`` (via ``_derive_side``).
    ``P_OVER`` / ``P_UNDER`` are kept so ``calibration()`` still works.
    """
    return {
        "DATE":             full.get("DATE"),
        "PLAYER_NAME":      full.get("PLAYER_NAME"),
        "MARKET":           full.get("MARKET"),
        "LINE_BOOKMAKER":   full.get("LINE_BOOKMAKER"),
        "LINE":             full.get("LINE"),
        "MIN_Q50":          full.get("MIN_Q50"),
        "STAT_Q50":         full.get("STAT_Q50"),
        "P_OVER":           full.get("P_OVER"),
        "P_UNDER":          full.get("P_UNDER"),
        "SIDE":             _derive_side(full),
        "ACTUAL_STAT":      full.get("ACTUAL_STAT"),
        "ACTUAL_MIN":       full.get("ACTUAL_MIN"),
        "HIT":              full.get("HIT"),
        "MISS_REASON":      full.get("MISS_REASON"),
        "ACTUAL_GAME_PTS":  full.get("ACTUAL_GAME_PTS"),
        "RECON_CONTEXT":    full.get("RECON_CONTEXT"),
        "RECONCILED_AT":    full.get("RECONCILED_AT"),
    }


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

    player_game = _player_game_rows(game_day, name)

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

def track_dfs_enriched(
    enriched_dir: str | Path = DFS_ENRICHED_DIR,
    *,
    run_timestamp: str | None = None,
) -> int:
    """Track every ``dfs_enriched_*.json`` pick into ``dfs_enriched.jsonl``."""
    return _track_dfs_pick_files(
        "dfs_enriched_*.json",
        DFS_ENRICHED_FILE,
        enriched_dir=enriched_dir,
        run_timestamp=run_timestamp,
    )


def track_dfs_sharp_aligned(
    enriched_dir: str | Path = DFS_ENRICHED_DIR,
    *,
    run_timestamp: str | None = None,
) -> int:
    """Track every ``dfs_sharp_aligned_*.json`` pick into ``dfs_sharp_aligned.jsonl``."""
    return _track_dfs_pick_files(
        "dfs_sharp_aligned_*.json",
        DFS_SHARP_ALIGNED_FILE,
        enriched_dir=enriched_dir,
        run_timestamp=run_timestamp,
    )


def track_all_slates(
    slates_dir: str | Path = SLATES_DIR,
    *,
    date: str | None = None,
    run_timestamp: str | None = None,
) -> int:
    """Track all current slate JSONs in ``data/props/ev_analysis`` into ``slates.jsonl``."""
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")
    if run_timestamp is None:
        run_timestamp = datetime.now().isoformat(timespec="seconds")

    rows: list[dict] = []
    for path in sorted(Path(slates_dir).expanduser().glob("*.json")):
        rows.extend(_slate_rows_from_file(path, date=date, run_timestamp=run_timestamp))

    if not rows:
        print("[log] slates       no data to log")
        return 0

    slate_df = pd.DataFrame(rows)
    _dedup_and_write(slate_df, SLATES_FILE, _SLATE_KEY)
    print(f"[log] slates       {len(rows):>4} slate rows  ({date})")
    return len(rows)


def track_all(
    *,
    enriched_dir: str | Path = DFS_ENRICHED_DIR,
    slates_dir: str | Path = SLATES_DIR,
    date: str | None = None,
    run_timestamp: str | None = None,
) -> dict[str, int]:
    """Track enriched picks, sharp-aligned picks, and slate legs with ledger deduping."""
    return {
        "dfs_enriched": track_dfs_enriched(enriched_dir, run_timestamp=run_timestamp),
        "dfs_sharp_aligned": track_dfs_sharp_aligned(enriched_dir, run_timestamp=run_timestamp),
        "slates": track_all_slates(slates_dir, date=date, run_timestamp=run_timestamp),
    }


def load_dfs_enriched() -> pd.DataFrame:
    """Load the full DFS enriched picks ledger."""
    return _read_jsonl(DFS_ENRICHED_FILE)


def load_dfs_sharp_aligned() -> pd.DataFrame:
    """Load the full DFS sharp-aligned picks ledger."""
    return _read_jsonl(DFS_SHARP_ALIGNED_FILE)


def load_predictions() -> pd.DataFrame:
    """Load the full predictions ledger."""
    return _read_jsonl(PREDICTIONS_FILE)


def load_slates() -> pd.DataFrame:
    """Load the full slates ledger (one row per slate, with attached legs)."""
    return _read_jsonl(SLATES_FILE)


def load_results() -> pd.DataFrame:
    """Load the full results ledger."""
    return _read_jsonl(RESULTS_FILE)


def results_compact_view(
    df: pd.DataFrame | None = None,
    *,
    date: str | None = None,
) -> pd.DataFrame:
    """
    Notebook-friendly slice of reconcile output (~9 columns).

    ``side`` is derived from ``STAT_Q50`` vs ``LINE`` (and P_OVER/P_UNDER at equality),
    same rule as the logging pipeline — not re-read from the stored ``SIDE`` column.

    Examples::

        log.results_compact_view(date="2026-05-01")
        log.results_compact_view(log.reconcile("2026-05-01", base_df))
    """
    if df is None:
        df = load_results()
    if df.empty:
        return pd.DataFrame()
    if date is not None and "DATE" in df.columns:
        df = df[df["DATE"] == date].copy()
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["side"] = out.apply(_derive_side, axis=1)

    want = [
        "PLAYER_NAME",
        "MARKET",
        "LINE",
        "MIN_Q50",
        "STAT_Q50",
        "side",
        "ACTUAL_STAT",
        "HIT",
        "MISS_REASON",
    ]
    have = [c for c in want if c in out.columns]
    slim = out[have].rename(
        columns={
            "PLAYER_NAME": "name",
            "MARKET": "market",
            "LINE": "line",
            "MIN_Q50": "min_q50",
            "STAT_Q50": "stat_q50",
            "ACTUAL_STAT": "actual_stat",
            "HIT": "hit",
            "MISS_REASON": "miss_reason",
        }
    )
    if "name" in slim.columns:
        slim = slim.sort_values(["name", "market", "line"], na_position="last")
    return slim.reset_index(drop=True)


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
    The slate ledger is stored as one row per slate; this helper expands ``LEGS``
    back to one row per leg with HIT and ACTUAL_STAT filled in.
    Group by SLATE_ID to check full parlay wins:

        sr = slate_results()
        parlay_wins = sr.groupby('SLATE_ID')['HIT'].all()
    """
    slates  = load_slates()
    results = load_results()
    if slates.empty or results.empty:
        return pd.DataFrame()

    leg_rows: list[dict] = []
    if "LEGS" in slates.columns:
        for _, slate in slates.iterrows():
            legs = slate.get("LEGS") or []
            if not isinstance(legs, list):
                continue
            slate_meta = {
                c: slate.get(c)
                for c in [
                    "DATE",
                    "RUN_TIMESTAMP",
                    "SLATE_ID",
                    "SLATE_TYPE",
                    "LINE_BOOKMAKER",
                    "PARLAY_PROB",
                    "EV",
                    "EV_DOLLARS",
                    "KELLY",
                    "PARLAY_N_LEGS",
                    "STRATEGY_TIER",
                    "COMBO_PROFILE",
                    "PLAYER_NAMES",
                ]
                if c in slates.columns
            }
            for leg_idx, leg in enumerate(legs, start=1):
                if not isinstance(leg, dict):
                    continue
                leg_rows.append({
                    **slate_meta,
                    "LEG_INDEX": leg_idx,
                    **leg,
                })
        slates = pd.DataFrame(leg_rows)
        if slates.empty:
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