"""
PrizePicks NBA projections — fetch and save a flat JSON export.

One HTTP parse, connection reuse via Session. Writes a single file:
  data/props/prizepicks/prizepicks_YYYY-MM-DD_HHMMSS.json

File schema: source, league, fetched_at, count, projections[]
Each projection: player, stat_type, line_score, odds_type, updated_at
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Project root (parent of src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_PROJECTIONS_DIR = os.path.join(_ROOT, "data", "props", "prizepicks")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")

API_URL = "https://api.prizepicks.com/projections?league_id=7"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://app.prizepicks.com/",
    "Origin": "https://app.prizepicks.com",
}

# Reused across requests in a process (typical single GET; still cheap to share)
_SESSION = requests.Session()
_SESSION.headers.update(HEADERS)

ProjectionRow = tuple[str, str, Any, str, str]
"""player, stat_type, line_score, odds_type, updated_at"""


def _prizepicks_output_filename() -> str:
    d = datetime.now(_OUTPUT_TZ)
    return d.strftime("prizepicks_%Y-%m-%d_%H%M%S.json")


def _resolve_prizepicks_output_path() -> str:
    """
    data/props/prizepicks/prizepicks_YYYY-MM-DD_HHMMSS.json under project root.
    PRIZEPICKS_OUTPUT=/abs/path/file.json overrides.
    """
    out = os.environ.get("PRIZEPICKS_OUTPUT", "").strip()
    if out and out.lower().endswith(".json"):
        expanded = os.path.expanduser(out)
        if not expanded.endswith(("/", "\\")) and not os.path.isdir(expanded):
            return expanded
    return os.path.join(_DEFAULT_PROJECTIONS_DIR, _prizepicks_output_filename())


def build_player_lookup(data: dict[str, Any]) -> dict[str, str]:
    """id -> name for type new_player (single pass over included)."""
    out: dict[str, str] = {}
    for elem in data.get("included") or []:
        if elem.get("type") != "new_player":
            continue
        eid = elem.get("id")
        if eid is None:
            continue
        name = (elem.get("attributes") or {}).get("name")
        if name is not None:
            out[eid] = name
    return out


def extract_projection_rows(data: dict[str, Any]) -> list[ProjectionRow]:
    """
    Flatten projections to (player, stat_type, line_score, odds_type, updated_at).
    Uses API strings for updated_at (no timezone parsing).
    """
    player_names = build_player_lookup(data)
    rows: list[ProjectionRow] = []
    for proj in data.get("data") or []:
        if proj.get("type") != "projection":
            continue
        attrs = proj.get("attributes") or {}
        line_score = attrs.get("line_score")
        if line_score is None:
            continue

        rel = proj.get("relationships") or {}
        np_ref = (rel.get("new_player") or {}).get("data") or {}
        pid = np_ref.get("id")
        player = player_names.get(pid) if pid is not None else ""
        if not player:
            player = attrs.get("description") or ""

        rows.append(
            (
                player,
                attrs.get("stat_type") or "",
                line_score,
                attrs.get("odds_type") or "",
                attrs.get("updated_at") or "",
            )
        )
    return rows


def fetch_projections_payload() -> dict[str, Any]:
    r = _SESSION.get(API_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def build_export_payload(
    rows: list[ProjectionRow],
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Flat, JSON-serializable export for modeling / review."""
    return {
        "source": "PrizePicks",
        "league": "NBA",
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_snapshot": os.path.basename(source_path) if source_path else None,
        "count": len(rows),
        "projections": [
            {
                "player": r[0],
                "stat_type": r[1],
                "line_score": r[2],
                "odds_type": r[3],
                "updated_at": r[4],
            }
            for r in rows
        ],
    }


def save_export(rows: list[ProjectionRow], out_path: str, *, source_path: str | None = None) -> None:
    payload = build_export_payload(rows, source_path=source_path)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_payload(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_rows_from_file(path: str) -> list[ProjectionRow]:
    """
    Rows from either our flat export (projections[]) or legacy PrizePicks API JSON (data[] + included[]).
    """
    obj = load_payload(path)
    if not isinstance(obj, dict):
        return []
    projs = obj.get("projections")
    if isinstance(projs, list) and projs and isinstance(projs[0], dict):
        if projs[0].get("type") != "projection" and "player" in projs[0]:
            return [
                (
                    p.get("player") or "",
                    p.get("stat_type") or "",
                    p.get("line_score"),
                    p.get("odds_type") or "",
                    p.get("updated_at") or "",
                )
                for p in projs
            ]
    return extract_projection_rows(obj)


def print_rows_tsv(rows: list[ProjectionRow], limit: int | None = None) -> None:
    """TSV to stdout; limit=None prints all."""
    print("player\tstat_type\tline_score\todds_type\tupdated_at")
    slice_rows = rows if limit is None else rows[:limit]
    for row in slice_rows:
        print("\t".join("" if x is None else str(x) for x in row))


class PrizePicks_Scraper:
    def __init__(self) -> None:
        self.lines: list[ProjectionRow] = []
        self.directory = _resolve_prizepicks_output_path()

        print(f"Output path: {self.directory}")
        data = fetch_projections_payload()
        self.lines = extract_projection_rows(data)
        save_export(self.lines, self.directory, source_path=None)
        print(f"Saved {len(self.lines)} projections -> {self.directory}")

    def loadJSON(self, path: str | None = None) -> None:
        """Load rows from a flat export or legacy raw API JSON on disk."""
        p = path or self.directory
        self.lines = load_rows_from_file(p)
        print(f"Loaded {len(self.lines)} projection rows from {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PrizePicks NBA projections")
    parser.add_argument(
        "--head",
        type=int,
        default=12,
        metavar="N",
        help="Print first N rows as TSV after fetch (default 12). Use 0 to skip.",
    )
    parser.add_argument(
        "--tsv-all",
        action="store_true",
        help="Print all rows as TSV (large).",
    )
    parser.add_argument(
        "--from-file",
        metavar="PATH",
        help="Skip network; load JSON and print rows only.",
    )
    parser.add_argument(
        "--save-export",
        action="store_true",
        help="With --from-file: also write a timestamped flat JSON via PRIZEPICKS_OUTPUT / default dir.",
    )
    args = parser.parse_args()

    if args.from_file:
        raw = os.path.expanduser(args.from_file)
        rows = load_rows_from_file(raw)
        print(f"Rows from file: {len(rows)}")
        if args.save_export:
            out = _resolve_prizepicks_output_path()
            save_export(rows, out, source_path=raw)
            print(f"Wrote flat JSON -> {out}")
        if args.tsv_all:
            print_rows_tsv(rows, limit=None)
        elif args.head > 0:
            print_rows_tsv(rows, limit=args.head)
    else:
        print("Starting PrizePicks fetch...")
        scraper = PrizePicks_Scraper()
        if args.tsv_all:
            print_rows_tsv(scraper.lines, limit=None)
        elif args.head > 0:
            print(f"\nSample (first {args.head} rows, TSV):")
            print_rows_tsv(scraper.lines, limit=args.head)
