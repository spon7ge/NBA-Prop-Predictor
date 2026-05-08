"""
Underdog Fantasy pick'em lines — fast fetch, slim JSON export.

Hot path: one GET, one json parse, no pandas. Saves pretty-printed JSON with
only: full_name, stat_name, stat_value, updated_at, choice, american_price, payout_multiplier.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_UNDERDOG_DIR = os.path.join(_ROOT, "data", "props", "underdogs")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")
_UNDERDOG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_DEFAULT_UNDERDOG_CONFIG: dict[str, Any] = {
    "ud_pickem_url": "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://app.underdogfantasy.com/",
    },
}

_CHOICE_MAP = {"lower": "under", "higher": "over"}

_SESSION = requests.Session()


def _underdog_output_filename() -> str:
    d = datetime.now(_OUTPUT_TZ)
    return d.strftime("underdog_%Y-%m-%d_%H%M%S.json")


def _resolve_underdog_output_path() -> str:
    """
    data/props/underdogs/underdog_YYYY-MM-DD_HHMMSS.json.
    UNDERDOG_OUTPUT=/abs/path/file.json overrides.
    """
    out = os.environ.get("UNDERDOG_OUTPUT", "").strip()
    if out and out.lower().endswith(".json"):
        expanded = os.path.expanduser(out)
        if not expanded.endswith(("/", "\\")) and not os.path.isdir(expanded):
            return expanded
    return os.path.join(_DEFAULT_UNDERDOG_DIR, _underdog_output_filename())


def _load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {**_DEFAULT_UNDERDOG_CONFIG}
    if os.path.isfile(_UNDERDOG_CONFIG_PATH):
        with open(_UNDERDOG_CONFIG_PATH, encoding="utf-8-sig") as f:
            user_cfg = json.load(f)
        base_headers = dict(_DEFAULT_UNDERDOG_CONFIG["headers"])
        if "headers" in user_cfg:
            base_headers.update(user_cfg["headers"])
        cfg.update(user_cfg)
        cfg["headers"] = base_headers
    return cfg


def _appearance_full_names(
    players: list[dict[str, Any]],
    appearances: list[dict[str, Any]],
    corrections: dict[str, str],
) -> dict[str, str]:
    """appearance_id -> display full name."""
    by_triple: dict[tuple[str, str, str], dict[str, Any]] = {}
    for p in players:
        try:
            key = (p["id"], p["position_id"], p["team_id"])
        except KeyError:
            continue
        by_triple[key] = p

    out: dict[str, str] = {}
    for a in appearances:
        try:
            key = (a["player_id"], a["position_id"], a["team_id"])
        except KeyError:
            continue
        p = by_triple.get(key)
        if not p:
            continue
        try:
            raw = f"{p['first_name']} {p['last_name']}"
        except KeyError:
            continue
        out[a["id"]] = corrections.get(raw, raw)
    return out


def extract_pick_rows(payload: dict[str, Any], corrections: dict[str, str]) -> list[dict[str, Any]]:
    """
    Flatten over_under_lines × options into pick dicts (seven public fields).
    Skips suspended lines/options (same intent as prior filter_data).
    """
    players = payload.get("players") or []
    appearances = payload.get("appearances") or []
    names = _appearance_full_names(players, appearances, corrections)

    rows: list[dict[str, Any]] = []
    for line in payload.get("over_under_lines") or []:
        if line.get("status") == "suspended":
            continue

        ou = line.get("over_under")
        if not isinstance(ou, dict):
            continue
        ast = ou.get("appearance_stat")
        if not isinstance(ast, dict):
            continue
        stat_name = ast.get("stat") or ""
        appearance_id_stat = ast.get("appearance_id")
        line_updated = line.get("updated_at")
        stat_value = line.get("stat_value")

        for opt in line.get("options") or []:
            if not isinstance(opt, dict):
                continue
            if opt.get("status") == "suspended":
                continue

            aid = opt.get("appearance_id") or appearance_id_stat
            if not aid:
                continue

            choice_raw = opt.get("choice")
            choice = _CHOICE_MAP.get(str(choice_raw).lower(), choice_raw)

            updated_at = opt.get("updated_at") or line_updated

            rows.append(
                {
                    "full_name": names.get(aid, ""),
                    "stat_name": stat_name,
                    "stat_value": stat_value,
                    "updated_at": updated_at,
                    "choice": choice,
                    "american_price": opt.get("american_price"),
                    "payout_multiplier": opt.get("payout_multiplier"),
                }
            )

    return rows


def build_export(picks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source": "Underdog Fantasy",
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(picks),
        "picks": picks,
    }


def save_export(picks: list[dict[str, Any]], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_export(picks), f, ensure_ascii=False, indent=2)


class UnderdogScraper:
    """Fetch pick'em payload and write a slim JSON file."""

    NAME_CORRECTIONS: dict[str, str] = {}

    def __init__(self) -> None:
        self.config = _load_config()
        self.underdog_props: list[dict[str, Any]] | None = None
        self.directory = _resolve_underdog_output_path()
        _SESSION.headers.update(self.config["headers"])
        print(f"Output path: {self.directory}")

    def fetch_data(self) -> dict[str, Any]:
        url = self.config["ud_pickem_url"]
        r = _SESSION.get(url, timeout=60)
        r.raise_for_status()
        return r.json()

    def scrape(self) -> None:
        payload = self.fetch_data()
        picks = extract_pick_rows(payload, self.NAME_CORRECTIONS)
        self.underdog_props = picks
        save_export(picks, self.directory)
        print(f"Saved {len(picks)} picks -> {self.directory}")


if __name__ == "__main__":
    print("Starting Underdog scraper...")
    try:
        scraper = UnderdogScraper()
        scraper.scrape()
        n = len(scraper.underdog_props) if scraper.underdog_props is not None else 0
        print(f"Done. {n} picks.")
    except Exception as e:
        print(f"Error: {e}")
        raise
