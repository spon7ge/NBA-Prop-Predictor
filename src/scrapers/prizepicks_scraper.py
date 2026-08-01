"""
PrizePicks NBA + WNBA projections — refactored for debuggability.

Fetches NBA and WNBA projections and exports clean JSON.
Handles DataDome (and legacy PerimeterX) via curl_cffi, requests, then Playwright.

Install for better success rate:
    pip install curl_cffi
    pip install playwright && playwright install chromium

Optional environment variables:
    PRIZEPICKS_OUTPUT: Override default output path (must end in .json)
    PRIZEPICKS_COOKIE: Browser session cookie (from DevTools)
    PRIZEPICKS_COOKIE_FILE: Path to file containing cookie
    PRIZEPICKS_IMPERSONATE: Comma-separated curl_cffi profiles (default: safari17_2_ios,...)
    PRIZEPICKS_HEADLESS: Set to 1/true/yes for headless Playwright (default: headed)
    LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)

Default output (one file per league):
    data/props/prizepicks/prizepicks_{league}_YYYY-MM-DD_HHMMSS.json
    e.g. prizepicks_wnba_2026-07-31_180238.json

Usage:
    python prizepicks_scraper.py                    # Fetch and save
    python prizepicks_scraper.py --head 20          # Fetch and show first 20
    python prizepicks_scraper.py --from-file data.json --head 10  # Load and show
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
_DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "data", "props", "prizepicks")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")

# PrizePicks league_id values (NBA=7, WNBA=3)
DEFAULT_LEAGUES: tuple[tuple[str, int], ...] = (("NBA", 7), ("WNBA", 3))
API_BASE = "https://api.prizepicks.com/projections"


def projections_api_url(league_id: int) -> str:
    """Build projections URL for a PrizePicks league_id."""
    return f"{API_BASE}?league_id={league_id}&per_page=250&single_stat=true"


# Back-compat alias (NBA)
API_URL = projections_api_url(7)

# Headers for requests library (no TLS impersonation)
HEADERS_REQUESTS: dict[str, str] = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

# Base headers for curl_cffi (impersonate profile provides User-Agent/sec-ch-ua)
HEADERS_CURL_BASE: dict[str, str] = {
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

# curl_cffi browser impersonation profiles to try (in order)
CURL_IMPERSONATE_DEFAULT = (
    "safari17_2_ios",  # Often works against PerimeterX
    "safari17_0",
    "chrome136",
    "chrome131",
    "chrome124",
    "chrome120",
)

# Origin variants to try (sometimes bot protection checks Origin header)
ORIGIN_VARIANTS = [
    {"name": "prizepicks.com", "Origin": "https://app.prizepicks.com", "Referer": "https://app.prizepicks.com/"},
    {"name": "www.prizepicks.com", "Origin": "https://www.prizepicks.com", "Referer": "https://www.prizepicks.com/"},
]

APP_URL = "https://app.prizepicks.com/"
PLAYWRIGHT_WAIT_SECONDS = 120
PLAYWRIGHT_POLL_INTERVAL_SECONDS = 2.0

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
class Projection:
    """A single player projection."""
    player: str
    stat_type: str
    line_score: float | int | None
    odds_type: str
    updated_at: str
    league: str = "NBA"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ExportData:
    """Complete export payload."""
    source: str = "PrizePicks"
    league: str = "NBA+WNBA"
    fetched_at: str = ""
    raw_snapshot: str | None = None
    count: int = 0
    projections: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.projections is None:
            self.projections = []
        self.count = len(self.projections) if self.projections else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "league": self.league,
            "fetched_at": self.fetched_at,
            "raw_snapshot": self.raw_snapshot,
            "count": self.count,
            "projections": self.projections or [],
        }


# ============================================================================
# Utility Functions
# ============================================================================

def league_slug(league: str) -> str:
    """Normalize league label for filenames (e.g. WNBA → wnba)."""
    return (league or "mixed").strip().lower().replace("+", "_").replace(" ", "_")


def resolve_output_path(
    league: str | None = None,
    *,
    when: datetime | None = None,
) -> str:
    """
    Resolve output file path.

    Checks:
    1. PRIZEPICKS_OUTPUT env var (if .json file) — used as-is when set
    2. Default: data/props/prizepicks/prizepicks_{league}_YYYY-MM-DD_HHMMSS.json
       e.g. prizepicks_wnba_2026-07-31_180238.json

    Args:
        league: League label embedded in the default filename (nba / wnba)
        when: Timestamp for the filename (defaults to now in America/Los_Angeles)

    Returns:
        Absolute path to output file
    """
    env_path = os.environ.get("PRIZEPICKS_OUTPUT", "").strip()

    if env_path and env_path.lower().endswith(".json"):
        expanded = os.path.expanduser(env_path)
        if not expanded.endswith(("/", "\\")) and not os.path.isdir(expanded):
            # When saving per-league under an explicit path, insert slug before .json
            if league:
                root, ext = os.path.splitext(expanded)
                slug = league_slug(league)
                if not root.lower().endswith(f"_{slug}"):
                    expanded = f"{root}_{slug}{ext}"
            logger.info(f"Using PRIZEPICKS_OUTPUT: {expanded}")
            return expanded

    stamp = when or datetime.now(_OUTPUT_TZ)
    slug = league_slug(league) if league else "mixed"
    filename = stamp.strftime(f"prizepicks_{slug}_%Y-%m-%d_%H%M%S.json")
    path = os.path.join(_DEFAULT_OUTPUT_DIR, filename)

    logger.info(f"Using default output path: {path}")
    return path


def get_cookie_for_request() -> str:
    """
    Load optional browser cookie.
    
    Checks (in order):
    1. PRIZEPICKS_COOKIE env var
    2. PRIZEPICKS_COOKIE_FILE env var
    
    Returns:
        Cookie string, or empty string if none found
    """
    # Try env var first
    c = os.environ.get("PRIZEPICKS_COOKIE", "").strip()
    if c:
        logger.debug("Using PRIZEPICKS_COOKIE from environment")
        return c
    
    # Try file
    path = os.environ.get("PRIZEPICKS_COOKIE_FILE", "").strip()
    if path:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded):
            logger.debug(f"Loading cookie from: {expanded}")
            try:
                with open(expanded, encoding="utf-8") as f:
                    return f.read().strip()
            except IOError as e:
                logger.warning(f"Failed to read cookie file: {e}")
                return ""
    
    logger.debug("No cookie found (PRIZEPICKS_COOKIE* not set)")
    return ""


def add_headers_to_dict(base: dict[str, str], overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Merge base headers with optional overrides and cookie."""
    h = dict(base)
    
    cookie = get_cookie_for_request()
    if cookie:
        h["Cookie"] = cookie
    
    if overrides:
        h.update(overrides)
    
    return h


def is_perimeterx_challenge(body: str) -> bool:
    """Check if response body indicates a PerimeterX challenge."""
    if not body:
        return False
    return any(marker in body for marker in ["px-cloud", '"appId":"PX', "PerimeterX"])


def is_datadome_challenge(
    body: str,
    headers: dict[str, str] | None = None,
) -> bool:
    """Check if response indicates a DataDome challenge."""
    if headers:
        for key, value in headers.items():
            if key.lower() == "x-datadome" and value:
                return True
    if not body:
        return False
    lowered = body.lower()
    return any(
        marker in lowered
        for marker in (
            "captcha-delivery.com",
            "geo.captcha-delivery.com",
            "datadome",
        )
    )


def is_bot_challenge(
    body: str,
    headers: dict[str, str] | None = None,
) -> bool:
    """True if response looks like DataDome or PerimeterX bot protection."""
    return is_datadome_challenge(body, headers) or is_perimeterx_challenge(body)


def headless_from_env() -> bool:
    """Return True when PRIZEPICKS_HEADLESS is a truthy value (default: headed)."""
    raw = os.environ.get("PRIZEPICKS_HEADLESS", "").strip().lower()
    return raw in {"1", "true", "yes"}


def build_fetch_failure_message() -> str:
    """Human-readable troubleshooting when all fetch paths fail."""
    return (
        "Failed to fetch PrizePicks projections.\n"
        "\n"
        "DataDome is blocking automated HTTP clients.\n"
        "\n"
        "Troubleshooting:\n"
        "1. Browser fallback (recommended when HTTP returns 403):\n"
        "   pip install playwright && playwright install chromium\n"
        "   Re-run headed (default) and solve any captcha in the window.\n"
        "   Use PRIZEPICKS_HEADLESS=1 only when no interactive captcha is expected.\n"
        "2. Optional browser session cookie for the HTTP path:\n"
        "   PRIZEPICKS_COOKIE='...' python prizepicks_scraper.py\n"
        "   Or: PRIZEPICKS_COOKIE_FILE=~/.prizepicks_cookie python ...\n"
        "3. curl_cffi impersonation profiles:\n"
        "   pip install curl_cffi\n"
        "   PRIZEPICKS_IMPERSONATE='safari17_0,chrome131' python ...\n"
        "\n"
        "Run with LOG_LEVEL=DEBUG for detailed diagnostics."
    )


def _normalize_response_headers(headers: Any) -> dict[str, str]:
    """Best-effort convert response header mapping to str→str."""
    if not headers:
        return {}
    try:
        return {str(k): str(v) for k, v in dict(headers).items()}
    except Exception:
        return {}


def _log_non_success_response(status_code: int, body: str, headers: Any) -> None:
    """Log why an HTTP response was rejected (DataDome / status)."""
    hdrs = _normalize_response_headers(headers)
    if is_datadome_challenge(body, hdrs):
        logger.debug(f"    → Status {status_code}, DataDome challenge")
    elif is_perimeterx_challenge(body):
        logger.debug(f"    → Status {status_code}, PerimeterX challenge")
    elif status_code == 403:
        logger.debug(f"    → Status {status_code} Forbidden (bot protection)")
    else:
        logger.debug(f"    → Status {status_code}, skipping")


def get_curl_impersonate_list() -> tuple[str, ...]:
    """
    Get list of curl_cffi impersonation profiles to try.
    
    Returns:
        Tuple of profile names in order of preference
    """
    env_list = os.environ.get("PRIZEPICKS_IMPERSONATE", "").strip()
    
    if env_list:
        profiles = tuple(p.strip() for p in env_list.split(",") if p.strip())
        logger.info(f"Using custom impersonate profiles: {profiles}")
        return profiles
    
    logger.debug(f"Using default impersonate profiles: {CURL_IMPERSONATE_DEFAULT}")
    return CURL_IMPERSONATE_DEFAULT


# ============================================================================
# API Fetching
# ============================================================================

def _parse_projections_json(text: str) -> dict[str, Any] | None:
    """Parse API JSON; return dict only when it has a 'data' key."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "data" in data:
        return data
    return None


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    get_fn: Any,
    get_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Shared GET → JSON helper for curl_cffi / requests."""
    kwargs = dict(get_kwargs or {})
    response = get_fn(url, headers=headers, timeout=timeout, **kwargs)
    body = getattr(response, "text", "") or ""
    if response.status_code != 200:
        _log_non_success_response(
            response.status_code,
            body,
            getattr(response, "headers", None),
        )
        return None
    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.debug("    → Invalid JSON, skipping")
        return None
    if isinstance(data, dict) and "data" in data:
        return data
    if is_bot_challenge(body, _normalize_response_headers(getattr(response, "headers", None))):
        logger.debug("    → Bot challenge payload (no 'data' field), skipping")
    else:
        logger.debug("    → Response missing 'data' field, skipping")
    return None


def try_fetch_with_curl_cffi(
    leagues: tuple[tuple[str, int], ...] = DEFAULT_LEAGUES,
) -> dict[str, dict[str, Any]] | None:
    """
    Try to fetch projections for each league using curl_cffi browser impersonation.

    Returns:
        Mapping league name → API payload, or None if no league succeeded
    """
    try:
        import curl_cffi.requests as curl_requests  # type: ignore[import-untyped]
    except ImportError:
        logger.info("curl_cffi not installed — skipping browser impersonation")
        return None

    logger.info("Attempting fetch with curl_cffi (browser impersonation)...")

    profiles = get_curl_impersonate_list()
    cookie = get_cookie_for_request()

    for i, profile in enumerate(profiles, 1):
        for origin_var in ORIGIN_VARIANTS:
            origin_name = origin_var["name"]
            headers = add_headers_to_dict(HEADERS_CURL_BASE, {
                "Origin": origin_var["Origin"],
                "Referer": origin_var["Referer"],
            })

            logger.debug(
                f"  Try {i}/{len(profiles)}: profile={profile}, "
                f"origin={origin_name}, cookie={'yes' if cookie else 'no'}"
            )

            try:
                out: dict[str, dict[str, Any]] = {}
                for league_name, league_id in leagues:
                    url = projections_api_url(league_id)
                    data = _http_get_json(
                        url,
                        headers=headers,
                        timeout=45,
                        get_fn=curl_requests.get,
                        get_kwargs={"impersonate": profile},
                    )
                    if data is not None:
                        out[league_name] = data
                        logger.info(
                            f"✓ curl_cffi {league_name} ok "
                            f"(profile={profile}, rows={len(data.get('data') or [])})"
                        )
                if out:
                    return out
            except Exception as e:
                logger.debug(f"    → Error: {type(e).__name__}: {e}")
                continue

    logger.info("curl_cffi attempts exhausted or no valid response")
    return None


def try_fetch_with_requests(
    leagues: tuple[tuple[str, int], ...] = DEFAULT_LEAGUES,
) -> dict[str, dict[str, Any]] | None:
    """
    Try to fetch projections for each league using stdlib requests.

    Often fails against DataDome without a browser session cookie.

    Returns:
        Mapping league name → API payload, or None if no league succeeded
    """
    logger.info("Attempting fetch with requests (stdlib)...")

    cookie = get_cookie_for_request()

    for origin_var in ORIGIN_VARIANTS:
        origin_name = origin_var["name"]
        headers = add_headers_to_dict(HEADERS_REQUESTS, {
            "Origin": origin_var["Origin"],
            "Referer": origin_var["Referer"],
        })

        logger.debug(f"  Try: origin={origin_name}, cookie={'yes' if cookie else 'no'}")

        try:
            out: dict[str, dict[str, Any]] = {}
            for league_name, league_id in leagues:
                url = projections_api_url(league_id)
                data = _http_get_json(
                    url,
                    headers=headers,
                    timeout=30,
                    get_fn=requests.get,
                )
                if data is not None:
                    out[league_name] = data
                    logger.info(
                        f"✓ requests {league_name} ok "
                        f"(rows={len(data.get('data') or [])})"
                    )
            if out:
                return out
        except requests.RequestException as e:
            logger.debug(f"    → Request error: {type(e).__name__}: {e}")
            continue

    logger.info("requests attempts exhausted")
    return None


def _playwright_fetch_url(page: Any, api_url: str) -> tuple[int, str]:
    """In-page fetch; returns (status, body_text)."""
    result = page.evaluate(
        """async (apiUrl) => {
            try {
                const r = await fetch(apiUrl, {
                    credentials: 'include',
                    headers: { 'Accept': 'application/json' },
                });
                const text = await r.text();
                return { ok: r.ok, status: r.status, text };
            } catch (e) {
                return { ok: false, status: 0, text: String(e) };
            }
        }""",
        api_url,
    )
    return int(result.get("status") or 0), (result.get("text") or "")


def try_fetch_with_playwright(
    leagues: tuple[tuple[str, int], ...] = DEFAULT_LEAGUES,
) -> dict[str, dict[str, Any]] | None:
    """
    Last-resort fetch: open PrizePicks in Chromium, clear DataDome, fetch each league.

    Headed by default so a captcha can be solved manually. Set PRIZEPICKS_HEADLESS=1
    for non-interactive runs.

    Returns:
        Mapping league name → API payload, or None if Playwright is unavailable / times out.
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
    except ImportError:
        logger.info(
            "playwright not installed — skipping browser fallback "
            "(pip install playwright && playwright install chromium)"
        )
        return None

    headless = headless_from_env()
    logger.info(
        "Attempting fetch with Playwright "
        f"({'headless' if headless else 'headed'}; wait up to "
        f"{PLAYWRIGHT_WAIT_SECONDS}s)..."
    )
    if not headless:
        logger.info(
            "If a captcha appears, solve it in the browser window; "
            "the scraper will keep polling the API."
        )

    captcha_logged = False
    deadline = time.monotonic() + PLAYWRIGHT_WAIT_SECONDS
    league_urls = [(name, projections_api_url(lid)) for name, lid in leagues]
    probe_url = league_urls[0][1]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            try:
                context = browser.new_context()
                page = context.new_page()
                page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

                cleared = False
                while time.monotonic() < deadline:
                    page_content = ""
                    try:
                        page_content = page.content()
                    except Exception:
                        pass

                    if is_datadome_challenge(page_content) and not captcha_logged:
                        logger.info(
                            "DataDome captcha detected — solve it in the browser window"
                        )
                        captcha_logged = True

                    try:
                        status, text = _playwright_fetch_url(page, probe_url)
                    except Exception as e:
                        logger.debug(f"  Playwright evaluate error: {type(e).__name__}: {e}")
                        time.sleep(PLAYWRIGHT_POLL_INTERVAL_SECONDS)
                        continue

                    data = _parse_projections_json(text) if status == 200 else None
                    if data is not None:
                        cleared = True
                        break

                    if is_datadome_challenge(text) and not captcha_logged:
                        logger.info(
                            "DataDome captcha detected — solve it in the browser window"
                        )
                        captcha_logged = True
                    else:
                        logger.debug(
                            f"  Playwright poll: status={status}, "
                            f"datadome={is_datadome_challenge(text)}"
                        )

                    time.sleep(PLAYWRIGHT_POLL_INTERVAL_SECONDS)

                if not cleared:
                    logger.info(
                        "Playwright wait exhausted without a valid projections payload"
                    )
                    return None

                out: dict[str, dict[str, Any]] = {}
                for league_name, url in league_urls:
                    try:
                        status, text = _playwright_fetch_url(page, url)
                    except Exception as e:
                        logger.debug(
                            f"  Playwright {league_name} error: {type(e).__name__}: {e}"
                        )
                        continue
                    data = _parse_projections_json(text) if status == 200 else None
                    if data is None:
                        logger.warning(
                            f"Playwright {league_name} failed "
                            f"(status={status}, datadome={is_datadome_challenge(text)})"
                        )
                        continue
                    out[league_name] = data
                    logger.info(
                        f"✓ Playwright {league_name} ok "
                        f"(rows={len(data.get('data') or [])})"
                    )

                return out or None
            finally:
                browser.close()
    except Exception as e:
        logger.info(f"Playwright fallback failed: {type(e).__name__}: {e}")
        return None


def fetch_projections_payloads(
    leagues: tuple[tuple[str, int], ...] = DEFAULT_LEAGUES,
) -> dict[str, dict[str, Any]]:
    """
    Fetch NBA + WNBA projections from PrizePicks API.

    Strategy:
    1. Try curl_cffi with browser impersonation
    2. Fall back to requests (often needs cookie against DataDome)
    3. Fall back to Playwright (headed; solve captcha if needed)
    4. If all fail, provide helpful error message

    Returns:
        Mapping league name → API response with 'data' / 'included'

    Raises:
        requests.HTTPError: If all fetch attempts fail
    """
    logger.info("=" * 70)
    logger.info(
        "FETCHING PRIZEPICKS PROJECTIONS "
        f"({', '.join(f'{n}={i}' for n, i in leagues)})"
    )
    logger.info("=" * 70)

    result = try_fetch_with_curl_cffi(leagues)
    if result is not None:
        return result

    result = try_fetch_with_requests(leagues)
    if result is not None:
        return result

    result = try_fetch_with_playwright(leagues)
    if result is not None:
        return result

    logger.error("=" * 70)
    logger.error("✗ FETCH FAILED: All attempts exhausted")
    logger.error("=" * 70)

    raise requests.HTTPError(build_fetch_failure_message())


def fetch_projections_payload() -> dict[str, Any]:
    """
    Back-compat: fetch NBA payload only (first configured league).

    Prefer ``fetch_projections_payloads`` for NBA+WNBA.
    """
    payloads = fetch_projections_payloads((("NBA", 7),))
    return payloads["NBA"]


# ============================================================================
# Data Processing
# ============================================================================

def build_player_lookup(data: dict[str, Any]) -> dict[str, str]:
    """
    Build player ID → name mapping from API response.

    Args:
        data: API response dict

    Returns:
        Dict mapping player IDs to full names
    """
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

    logger.debug(f"Built player lookup: {len(out)} players")
    return out


def extract_projections(
    data: dict[str, Any],
    *,
    league: str = "NBA",
) -> list[Projection]:
    """
    Extract projections from API response.

    Flattens nested structure into clean Projection objects.

    Args:
        data: API response dict
        league: League label stamped onto each projection

    Returns:
        List of Projection objects
    """
    logger.info(f"Extracting {league} projections...")

    player_names = build_player_lookup(data)
    projections: list[Projection] = []

    raw_projections = data.get("data") or []
    logger.debug(f"Processing {len(raw_projections)} raw projections...")

    skipped_wrong_type = 0
    skipped_no_line_score = 0
    skipped_no_player = 0
    added = 0

    for proj in raw_projections:
        if proj.get("type") != "projection":
            skipped_wrong_type += 1
            continue

        attrs = proj.get("attributes") or {}
        line_score = attrs.get("line_score")

        if line_score is None:
            skipped_no_line_score += 1
            continue

        rel = proj.get("relationships") or {}
        np_ref = (rel.get("new_player") or {}).get("data") or {}
        pid = np_ref.get("id")

        player = ""
        if pid is not None:
            player = player_names.get(pid, "")

        if not player:
            player = attrs.get("description") or ""

        if not player:
            skipped_no_player += 1
            continue

        projection = Projection(
            player=player,
            stat_type=attrs.get("stat_type") or "",
            line_score=line_score,
            odds_type=attrs.get("odds_type") or "",
            updated_at=attrs.get("updated_at") or "",
            league=league,
        )
        projections.append(projection)
        added += 1

    logger.info(
        f"Extraction complete ({league}): added={added}, skipped_type={skipped_wrong_type}, "
        f"skipped_no_line={skipped_no_line_score}, skipped_no_player={skipped_no_player}"
    )

    return projections


def extract_projections_from_payloads(
    payloads: dict[str, dict[str, Any]],
) -> list[Projection]:
    """Extract and concatenate projections for each league payload."""
    rows: list[Projection] = []
    for league_name, payload in payloads.items():
        rows.extend(extract_projections(payload, league=league_name))
    return rows


# ============================================================================
# File I/O
# ============================================================================

def save_projections(
    projections: list[Projection],
    path: str,
    *,
    source_path: str | None = None,
    league_label: str | None = None,
) -> None:
    """
    Save projections to JSON file.

    Args:
        projections: List of Projection objects
        path: Output file path
        source_path: Optional path to source file (for raw_snapshot field)
        league_label: Top-level league string (default derived from rows)
    """
    logger.info(f"Saving {len(projections)} projections to {path}...")

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
        logger.debug(f"Ensured directory exists: {parent}")

    if league_label is None:
        leagues = sorted({p.league for p in projections if p.league})
        league_label = "+".join(leagues) if leagues else "NBA+WNBA"

    proj_dicts = [p.to_dict() for p in projections]
    export = ExportData(
        league=league_label,
        projections=proj_dicts,
        raw_snapshot=os.path.basename(source_path) if source_path else None,
    )

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved successfully to {path}")
    except IOError as e:
        logger.error(f"Failed to write file: {e}")
        raise


def load_projections_from_file(path: str) -> list[Projection]:
    """
    Load projections from either flat export or legacy API JSON.

    Args:
        path: File path

    Returns:
        List of Projection objects
    """
    logger.info(f"Loading projections from: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load file: {e}")
        raise

    if not isinstance(obj, dict):
        logger.warning("Loaded JSON is not a dict, returning empty list")
        return []

    projs = obj.get("projections")
    if isinstance(projs, list) and projs and isinstance(projs[0], dict):
        if "player" in projs[0]:
            logger.debug("Recognized as flat export format")
            default_league = obj.get("league") or "NBA"
            if isinstance(default_league, str) and "+" in default_league:
                default_league = "NBA"
            projections = [
                Projection(
                    player=p.get("player") or "",
                    stat_type=p.get("stat_type") or "",
                    line_score=p.get("line_score"),
                    odds_type=p.get("odds_type") or "",
                    updated_at=p.get("updated_at") or "",
                    league=p.get("league") or default_league,
                )
                for p in projs
            ]
            logger.info(f"Loaded {len(projections)} projections from flat export")
            return projections

    logger.debug("Treating as legacy API JSON format")
    league = obj.get("league") if isinstance(obj.get("league"), str) else "NBA"
    if "+" in str(league):
        league = "NBA"
    projections = extract_projections(obj, league=league)
    logger.info(f"Loaded {len(projections)} projections from API format")
    return projections


# ============================================================================
# Display
# ============================================================================

def print_projections_tsv(projections: list[Projection], limit: int | None = None) -> None:
    """
    Print projections as TSV to stdout.

    Args:
        projections: List of Projection objects
        limit: Maximum number to print (None = all)
    """
    print("league\tplayer\tstat_type\tline_score\todds_type\tupdated_at")

    slice_projs = projections if limit is None else projections[:limit]
    for proj in slice_projs:
        print(
            f"{proj.league}\t{proj.player}\t{proj.stat_type}\t"
            f"{proj.line_score}\t{proj.odds_type}\t{proj.updated_at}"
        )


# ============================================================================
# Main Scraper
# ============================================================================

class PrizePicks_Scraper:
    """Main scraper orchestrator."""

    def __init__(self) -> None:
        logger.info("Initializing PrizePicks_Scraper...")
        self.output_paths: list[str] = []
        self.projections: list[Projection] = []
        self.scraped_at: datetime | None = None
        logger.info("Initialization complete")

    @property
    def output_path(self) -> str:
        """Most recent output path (last league file written), for CLI --save-export."""
        if self.output_paths:
            return self.output_paths[-1]
        return resolve_output_path()

    def _save_by_league(self, projections: list[Projection]) -> list[str]:
        """Write one prizepicks_{league}_*.json file per league present."""
        when = datetime.now(_OUTPUT_TZ)
        by_league: dict[str, list[Projection]] = {}
        for row in projections:
            by_league.setdefault(row.league or "NBA", []).append(row)

        if not by_league:
            # No rows — still emit empty files for configured leagues
            for league_name, _ in DEFAULT_LEAGUES:
                by_league[league_name] = []

        scraped_at = self.scraped_at or datetime.now(timezone.utc)
        self.scraped_at = scraped_at

        paths: list[str] = []
        for league_name in sorted(by_league.keys()):
            rows = by_league[league_name]
            path = resolve_output_path(league_name, when=when)
            save_projections(rows, path, league_label=league_name)
            try:
                from src.odds.load_snapshots import load_prizepicks_snapshot

                n = load_prizepicks_snapshot(
                    [p.to_dict() for p in rows],
                    league=league_slug(league_name),
                    scraped_at=scraped_at,
                )
                logger.info(
                    f"Supabase odds.wnba_prizepicks upserted {n} rows ({league_name})"
                )
            except Exception as e:
                logger.error(f"Supabase prizepicks load failed (JSON kept): {e}")
            paths.append(path)
        self.output_paths = paths
        return paths

    def run(self) -> None:
        """Execute the full scrape pipeline for NBA + WNBA."""
        try:
            logger.info("\n[Step 1/2] Fetching data...")
            payloads = fetch_projections_payloads()

            logger.info("\n[Step 2/2] Extracting and saving...")
            self.projections = extract_projections_from_payloads(payloads)
            self.scraped_at = datetime.now(timezone.utc)
            paths = self._save_by_league(self.projections)

            by_league: dict[str, int] = {}
            for p in self.projections:
                by_league[p.league] = by_league.get(p.league, 0) + 1
            summary = ", ".join(f"{k}={v}" for k, v in sorted(by_league.items())) or "none"

            logger.info("=" * 70)
            logger.info(
                f"✓ SUCCESS: {len(self.projections)} projections saved ({summary})"
            )
            for path in paths:
                logger.info(f"  → {path}")
            logger.info("=" * 70)

        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"✗ FAILED: {type(e).__name__}: {e}")
            logger.error("=" * 70)
            raise

    def load_from_file(self, path: str) -> None:
        """Load projections from file."""
        try:
            expanded = os.path.expanduser(path)
            self.projections = load_projections_from_file(expanded)
            logger.info(f"Loaded {len(self.projections)} projections")
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="PrizePicks NBA + WNBA projections scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           Fetch and save projections
  %(prog)s --head 20                 Fetch and show first 20
  %(prog)s --from-file data.json     Load from file and show sample
  %(prog)s --from-file data.json --tsv-all  Load and show all as TSV

Environment variables:
  LOG_LEVEL                          DEBUG, INFO, WARNING, or ERROR
  PRIZEPICKS_COOKIE                  Browser session cookie
  PRIZEPICKS_COOKIE_FILE             Path to cookie file
  PRIZEPICKS_IMPERSONATE             Comma-separated curl_cffi profiles
  PRIZEPICKS_HEADLESS                1/true/yes for headless Playwright (default headed)
        """,
    )

    parser.add_argument(
        "--head",
        type=int,
        default=12,
        metavar="N",
        help="Print first N rows as TSV (default 12, use 0 to skip)",
    )
    parser.add_argument(
        "--tsv-all",
        action="store_true",
        help="Print all rows as TSV (can be large)",
    )
    parser.add_argument(
        "--from-file",
        metavar="PATH",
        help="Load projections from JSON file instead of fetching",
    )
    parser.add_argument(
        "--save-export",
        action="store_true",
        help="With --from-file: also save as timestamped export",
    )

    args = parser.parse_args()

    if args.from_file:
        scraper = PrizePicks_Scraper()
        scraper.load_from_file(args.from_file)

        if args.save_export:
            logger.info("\nSaving export...")
            scraper._save_by_league(scraper.projections)

        if args.tsv_all:
            print()
            print_projections_tsv(scraper.projections, limit=None)
        elif args.head > 0:
            print(f"\nShowing first {args.head} rows (TSV):")
            print()
            print_projections_tsv(scraper.projections, limit=args.head)

    else:
        logger.info("Starting PrizePicks scraper...")
        scraper = PrizePicks_Scraper()
        scraper.run()

        if args.tsv_all:
            print()
            print_projections_tsv(scraper.projections, limit=None)
        elif args.head > 0:
            print(f"\nShowing first {args.head} rows (TSV):")
            print()
            print_projections_tsv(scraper.projections, limit=args.head)


if __name__ == "__main__":
    main()
