"""
Underdog Fantasy pick'em scraper — refactored for debuggability.

Fetches NBA/WNBA pick'em lines from Underdog Fantasy API and exports clean JSON.
Includes detailed logging at each step to make debugging easy.

Usage:
    python underdog_scraper.py
    
Environment variables:
    UNDERDOG_OUTPUT: Override default output path (must end in .json);
                     sport slug is appended when saving multiple leagues
    UNDERDOG_URL: Override API endpoint
    LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

# ============================================================================
# Configuration
# ============================================================================

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "data", "props", "underdogs")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_DEFAULT_CONFIG: dict[str, Any] = {
    "sport_allowlist": ["NBA", "WNBA"],
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

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging with timestamp and level indicators."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="[%(levelname)-8s] %(name)s: %(message)s",
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Pick:
    """A single prop pick with all public fields."""
    full_name: str
    stat_name: str
    stat_value: float | None
    updated_at: str | None
    choice: str
    american_price: int | None
    payout_multiplier: float | None
    sport_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (omit internal sport_id)."""
        data = asdict(self)
        data.pop("sport_id", None)
        return data


@dataclass
class ExportData:
    """Complete export payload."""
    source: str = "Underdog Fantasy"
    fetched_at: str = ""
    count: int = 0
    picks: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.picks is None:
            self.picks = []
        self.count = len(self.picks) if self.picks else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "fetched_at": self.fetched_at,
            "count": self.count,
            "picks": self.picks or [],
        }


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config() -> dict[str, Any]:
    """Load and merge default + user config."""
    cfg = {**_DEFAULT_CONFIG}
    
    if os.path.isfile(_CONFIG_PATH):
        logger.info(f"Loading config from: {_CONFIG_PATH}")
        try:
            with open(_CONFIG_PATH, encoding="utf-8-sig") as f:
                user_cfg = json.load(f)
            logger.debug(f"User config keys: {list(user_cfg.keys())}")
            
            # Merge headers carefully
            if "headers" in user_cfg:
                cfg["headers"].update(user_cfg["headers"])
            
            # Merge other config
            cfg.update(user_cfg)
            logger.info("Config merged successfully")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
    else:
        logger.info(f"No config file found at {_CONFIG_PATH}. Using defaults.")
    
    return cfg


def get_sport_allowlist(cfg: dict[str, Any]) -> frozenset[str] | None:
    """
    Parse sport allowlist from config.
    
    Returns:
        Frozenset of sport IDs to keep, or None to disable filtering.
    """
    raw = cfg.get("sport_allowlist", ["NBA", "WNBA"])
    
    if raw is None:
        logger.info("Sport allowlist is None — will keep all sports")
        return None
    
    result = frozenset(str(x) for x in raw)
    logger.info(f"Sport allowlist: {result}")
    return result


# ============================================================================
# API Fetching
# ============================================================================

def fetch_underdogfantasy(url: str, headers: dict[str, str]) -> dict[str, Any]:
    """
    Fetch pick'em payload from Underdog Fantasy API.
    
    Args:
        url: API endpoint
        headers: HTTP headers to send
        
    Returns:
        Parsed JSON response
        
    Raises:
        requests.RequestException: On HTTP error
        ValueError: On invalid JSON
    """
    logger.info(f"Fetching from: {url}")
    
    try:
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Fetched successfully. Status: {response.status_code}")
        
        # Log payload shape
        games = len(data.get("games") or [])
        solo_games = len(data.get("solo_games") or [])
        appearances = len(data.get("appearances") or [])
        lines = len(data.get("over_under_lines") or [])
        
        logger.info(
            f"Payload shape: games={games}, solo_games={solo_games}, "
            f"appearances={appearances}, lines={lines}"
        )
        
        return data
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise
    except ValueError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise


# ============================================================================
# Data Processing
# ============================================================================

def build_appearance_sport_map(payload: dict[str, Any]) -> dict[str, str]:
    """
    Build appearance_id -> sport_id mapping.
    
    Args:
        payload: API response payload
        
    Returns:
        Dict mapping appearance IDs to sport IDs
    """
    games_by_id = {g["id"]: g for g in (payload.get("games") or []) if isinstance(g, dict)}
    solo_by_id = {g["id"]: g for g in (payload.get("solo_games") or []) if isinstance(g, dict)}
    
    logger.debug(f"Games by ID: {len(games_by_id)} entries")
    logger.debug(f"Solo games by ID: {len(solo_by_id)} entries")
    
    out: dict[str, str] = {}
    
    for a in payload.get("appearances") or []:
        if not isinstance(a, dict):
            continue
        
        aid = a.get("id")
        mid = a.get("match_id")
        
        if not (aid and mid):
            continue
        
        evt = games_by_id.get(mid) or solo_by_id.get(mid)
        if not isinstance(evt, dict):
            continue
        
        sid = evt.get("sport_id")
        if sid:
            out[str(aid)] = str(sid)
    
    logger.debug(f"Built appearance->sport map: {len(out)} entries")
    return out


def build_player_lookup(
    players: list[dict[str, Any]],
    appearances: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Build appearance_id -> full_name mapping.
    
    Args:
        players: Player records from payload
        appearances: Appearance records from payload
        
    Returns:
        Dict mapping appearance IDs to player full names
    """
    # Index players by (player_id, position_id, team_id) triple
    by_triple: dict[tuple[str, str, str], dict[str, Any]] = {}
    
    for p in players:
        try:
            key = (p["id"], p["position_id"], p["team_id"])
            by_triple[key] = p
        except (KeyError, TypeError):
            continue
    
    logger.debug(f"Indexed {len(by_triple)} players by (id, position, team)")
    
    # Match appearances to players and build names
    out: dict[str, str] = {}
    
    for a in appearances:
        try:
            key = (a["player_id"], a["position_id"], a["team_id"])
        except (KeyError, TypeError):
            continue
        
        p = by_triple.get(key)
        if not p:
            continue
        
        try:
            full_name = f"{p['first_name']} {p['last_name']}"
            out[a["id"]] = full_name
        except KeyError:
            continue
    
    logger.debug(f"Resolved {len(out)} appearance->name mappings")
    return out


def extract_picks(
    payload: dict[str, Any],
    sport_allowlist: frozenset[str] | None,
) -> list[Pick]:
    """
    Extract and flatten pick'em lines into individual picks.
    
    Args:
        payload: API response payload
        sport_allowlist: Set of sport IDs to keep (None = keep all)
        
    Returns:
        List of Pick objects
    """
    logger.info("Extracting picks from payload...")
    
    # Build lookups
    names = build_player_lookup(payload.get("players") or [], payload.get("appearances") or [])
    app_sports = build_appearance_sport_map(payload)
    
    picks: list[Pick] = []
    lines = payload.get("over_under_lines") or []
    
    logger.info(f"Processing {len(lines)} over/under lines...")
    
    suspended_lines = 0
    skipped_options = 0
    skipped_sport = 0
    added = 0
    
    for line in lines:
        # Skip suspended lines
        if line.get("status") == "suspended":
            suspended_lines += 1
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
        
        # Process options
        for opt in line.get("options") or []:
            if not isinstance(opt, dict):
                continue
            
            # Skip suspended options
            if opt.get("status") == "suspended":
                skipped_options += 1
                continue
            
            aid = opt.get("appearance_id") or appearance_id_stat
            if not aid:
                continue
            
            sid = app_sports.get(str(aid), "")

            # Filter by sport if allowlist is set
            if sport_allowlist is not None:
                if sid not in sport_allowlist:
                    skipped_sport += 1
                    continue
            
            # Map choice
            choice_raw = opt.get("choice")
            choice = _CHOICE_MAP.get(str(choice_raw).lower(), choice_raw)
            
            # Use option timestamp or fall back to line timestamp
            updated_at = opt.get("updated_at") or line_updated
            
            # Create pick
            pick = Pick(
                full_name=names.get(str(aid), ""),
                stat_name=stat_name,
                stat_value=stat_value,
                updated_at=updated_at,
                choice=choice,
                american_price=opt.get("american_price"),
                payout_multiplier=opt.get("payout_multiplier"),
                sport_id=sid,
            )
            picks.append(pick)
            added += 1
    
    logger.info(
        f"Extraction complete: added={added}, "
        f"suspended_lines={suspended_lines}, suspended_options={skipped_options}, "
        f"skipped_sport={skipped_sport}"
    )
    
    return picks


# ============================================================================
# File Export
# ============================================================================

def resolve_output_path(sport: str) -> str:
    """
    Resolve output file path for a single sport.
    
    Checks:
    1. UNDERDOG_OUTPUT env var (if .json file) — used as-is for a single sport
    2. Default: data/props/underdogs/underdog_{sport}_YYYY-MM-DD_HHMMSS.json
    
    Returns:
        Absolute path to output file
    """
    env_path = os.environ.get("UNDERDOG_OUTPUT", "").strip()
    sport_slug = sport.strip().lower() or "unknown"
    
    if env_path and env_path.lower().endswith(".json"):
        expanded = os.path.expanduser(env_path)
        if not expanded.endswith(("/", "\\")) and not os.path.isdir(expanded):
            # When env points at a single file and we have multiple sports, stamp sport in.
            root, ext = os.path.splitext(expanded)
            if sport_slug not in os.path.basename(root).lower():
                expanded = f"{root}_{sport_slug}{ext}"
            logger.info(f"Using UNDERDOG_OUTPUT: {expanded}")
            return expanded
    
    # Default path: underdog_wnba_YYYY-MM-DD_HHMMSS.json
    now = datetime.now(_OUTPUT_TZ)
    filename = now.strftime(f"underdog_{sport_slug}_%Y-%m-%d_%H%M%S.json")
    path = os.path.join(_DEFAULT_OUTPUT_DIR, filename)
    
    logger.info(f"Using default output path: {path}")
    return path


def group_picks_by_sport(picks: list[Pick]) -> dict[str, list[Pick]]:
    """Group picks by sport_id (empty → unknown)."""
    grouped: dict[str, list[Pick]] = {}
    for pick in picks:
        key = pick.sport_id or "unknown"
        grouped.setdefault(key, []).append(pick)
    return grouped


def sport_to_league(sport: str) -> str | None:
    """Map Underdog sport_id to odds league slug (nba / wnba only)."""
    normalized = sport.strip().upper()
    if normalized == "NBA":
        return "nba"
    if normalized == "WNBA":
        return "wnba"
    return None


def save_picks(picks: list[Pick], path: str) -> None:
    """
    Save picks to JSON file.
    
    Args:
        picks: List of Pick objects
        path: Output file path
    """
    logger.info(f"Saving {len(picks)} picks to {path}...")
    
    # Create parent directory
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
        logger.debug(f"Ensured directory exists: {parent}")
    
    # Build export
    pick_dicts = [p.to_dict() for p in picks]
    export = ExportData(picks=pick_dicts)
    
    # Write file
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved successfully to {path}")
    except IOError as e:
        logger.error(f"Failed to write file: {e}")
        raise


# ============================================================================
# Main Scraper
# ============================================================================

class UnderdogScraper:
    """Main scraper orchestrator."""
    
    def __init__(self) -> None:
        logger.info("Initializing UnderdogScraper...")
        self.config = load_config()
        self.output_paths: list[str] = []
        self.picks: list[Pick] = []
        self.scraped_at: datetime | None = None
        logger.info("Initialization complete")

    def _load_sport_to_supabase(self, sport: str, sport_picks: list[Pick]) -> None:
        """Upsert one sport batch to odds.wnba_underdogs; JSON save is already done."""
        league = sport_to_league(sport)
        if league is None:
            logger.info(f"Skipping Supabase load for unmapped sport {sport!r}")
            return
        try:
            from src.odds.load_snapshots import load_underdog_snapshot

            n = load_underdog_snapshot(
                [p.to_dict() for p in sport_picks],
                league=league,
                scraped_at=self.scraped_at,
            )
            logger.info(f"Supabase odds.wnba_underdogs upserted {n} rows ({sport})")
        except Exception as e:
            logger.error(f"Supabase underdog load failed (JSON kept): {e}")
    
    def run(self) -> None:
        """Execute the full scrape pipeline."""
        logger.info("=" * 70)
        logger.info("STARTING UNDERDOG SCRAPER")
        logger.info("=" * 70)
        
        try:
            # Step 1: Fetch
            logger.info("\n[Step 1/3] Fetching data...")
            url = self.config.get("ud_pickem_url")
            headers = self.config.get("headers", {})
            payload = fetch_underdogfantasy(url, headers)
            
            # Step 2: Extract
            logger.info("\n[Step 2/3] Extracting picks...")
            sport_allowlist = get_sport_allowlist(self.config)
            self.picks = extract_picks(payload, sport_allowlist)
            
            # Step 3: Save one file per sport (underdog_wnba_*, underdog_nba_*, …)
            logger.info("\n[Step 3/3] Saving to file...")
            self.scraped_at = datetime.now(timezone.utc)
            grouped = group_picks_by_sport(self.picks)
            self.output_paths = []
            if not grouped:
                # Still write an empty WNBA file when allowlist includes it, else first allowlist sport
                fallback = "WNBA"
                if sport_allowlist:
                    fallback = sorted(sport_allowlist)[0]
                path = resolve_output_path(fallback)
                save_picks([], path)
                self._load_sport_to_supabase(fallback, [])
                self.output_paths.append(path)
            else:
                for sport, sport_picks in sorted(grouped.items()):
                    path = resolve_output_path(sport)
                    save_picks(sport_picks, path)
                    self._load_sport_to_supabase(sport, sport_picks)
                    self.output_paths.append(path)
            
            logger.info("=" * 70)
            logger.info(
                f"✓ SUCCESS: {len(self.picks)} picks saved across "
                f"{len(self.output_paths)} file(s)"
            )
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"✗ FAILED: {type(e).__name__}: {e}")
            logger.error("=" * 70)
            raise


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    scraper = UnderdogScraper()
    scraper.run()