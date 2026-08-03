"""
Pinnacle player props scraper via Selenium (no Arcadia REST client).
Supports NBA and WNBA.

Flow:
  1) Open https://www.pinnacle.com/en/basketball/{league}/matchups/#all
  2) Collect each game URL like …/basketball/{league}/{slug}/{matchup_id}/#all
  3) Visit each game page; Arcadia responses are decoded from Chrome DevTools
     (responses the site loaded in the browser after navigation).

Requirements: Google Chrome + Selenium-managed chromedriver.
Optional PINNACLE_CHROME_VISIBLE=1 (disable headless if pages load oddly).

Each game's output includes BOTH player props ("props") AND game-level markets
("team_markets": moneyline / spread / total, grouped by period — 0 = full game).
PINNACLE_MAX_GAMES caps games per run, PER LEAGUE (default 50).
PINNACLE_PAGE_WAIT — fallback sleep between retries if Arcadia poll fails (default 8).
PINNACLE_ODDS_FORMAT=both|decimal|american — Arcadia always sends American `price`; decimal
is derived. The site odds-format switch does not change the API (no need to automate it).

PINNACLE_LEAGUES — comma-separated list of leagues to scrape, chosen from "nba", "wnba".
Default is "nba,wnba" (both). Example: PINNACLE_LEAGUES=wnba to scrape only WNBA.

Output: data/props/pinnacle/{league}/pinnacle_{league}_{YYYY-MM-DD}_{HHMMSS}.json
(America/Los_Angeles, PST/PDT). Override with:
  PINNACLE_OUTPUT     — full path to a .json file. If both leagues run in the same
                         invocation, the league is inserted before the extension
                         (e.g. "out.json" -> "out_nba.json", "out_wnba.json") so the
                         two runs don't clobber each other. A directory is used as-is
                         (timestamped per-league name is appended inside it).
  PINNACLE_OUTPUT_DIR — directory; timestamped per-league filename appended inside it.

PINNACLE_WORKERS — concurrent Chrome instances for game pages (default scales with CPU, max 6).
Discovery still uses one short-lived driver per league; each worker gets its own browser +
temp user-data-dir.

Speed-oriented (optional):
PINNACLE_EAGER_PAGE_LOAD is ignored: discovery always uses full page load; game pages use
eager by default (set PINNACLE_MATCHUP_NORMAL_LOAD=1 to force full load on games too).
PINNACLE_BLOCK_IMAGES=1 — block images on game pages only (not on discovery).
PINNACLE_ARCADIA_TIMEOUT — seconds to poll for Arcadia /matchups responses (default 90).
PINNACLE_ARCADIA_POLL — poll interval seconds (default 0.22).
PINNACLE_POST_ARCADIA_BUFFER — brief sleep after both payloads decode (default 0.08).
PINNACLE_DISCOVER_SCROLL_ROUNDS / PINNACLE_DISCOVER_SCROLL_PAUSE — list-page lazy load (8 / 0.32).
PINNACLE_DISCOVER_SETTLE — seconds to wait after #root before scrolling (default 0.65).
PINNACLE_DISCOVER_RETRIES / PINNACLE_DISCOVER_RETRY_PAUSE — empty list reloads (3 / 2.5s).
PINNACLE_GAMES_PER_WORKER — target games per browser when parallel (default 2); fewer
Chrome cold-starts on small slates (e.g. 4 games → 2 workers × 2 pages each).
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from selenium.webdriver.common.by import By  # type: ignore[import-untyped]
    from selenium.webdriver.support import expected_conditions as EC  # type: ignore[import-untyped]
    from selenium.webdriver.support.wait import WebDriverWait  # type: ignore[import-untyped]
except ImportError:
    By = EC = WebDriverWait = None  # type: ignore[assignment,misc]
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

# Always under repo unless overridden (do not use PROJECTIONS_DIRECTORY — that is for other books).
_DEFAULT_PINNACLE_BASE_DIR = os.path.join(_ROOT, "data", "props", "pinnacle")


_PINNACLE_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")

PINNACLE_ORIGIN = "https://www.pinnacle.com"

# Supported leagues -> their Pinnacle matchups list URL. Both leagues live under the
# basketball vertical with the same page shape and same Arcadia payload shape.
LEAGUE_MATCHUPS_URL: dict[str, str] = {
    "nba": "https://www.pinnacle.com/en/basketball/nba/matchups/#all",
    "wnba": "https://www.pinnacle.com/en/basketball/wnba/matchups/#all",
}
SUPPORTED_LEAGUES = tuple(LEAGUE_MATCHUPS_URL.keys())

# Related-market `units` -> stable stat key in JSON output (same semantics as Arcadia payloads,
# same for NBA and WNBA).
PLAYER_PROP_UNITS = {
    "Points": "points",
    "Assists": "assists",
    "Rebounds": "rebounds",
    "PointsReboundsAssist": "points_rebounds_assists",
}

# Team/game-level (non player-prop) market types pulled alongside props.
TEAM_MARKET_TYPES = ("moneyline", "spread", "total")

# Matches /en/basketball/{nba|wnba}/{slug}/{matchup_id}/
_GAME_PATH_RE = re.compile(
    r"/en/basketball/(?P<league>nba|wnba)/(?P<slug>[\w.-]+)/(?P<mid>\d+)/?",
    re.IGNORECASE,
)


def _pinnacle_output_filename(league: str, now: dt.datetime | None = None) -> str:
    """Filesystem-safe name: pinnacle_nba_2026-05-06_213045.json in Pacific time."""
    if now is None:
        d = dt.datetime.now(_PINNACLE_OUTPUT_TZ)
    elif now.tzinfo is None:
        d = now.replace(tzinfo=dt.timezone.utc).astimezone(_PINNACLE_OUTPUT_TZ)
    else:
        d = now.astimezone(_PINNACLE_OUTPUT_TZ)
    return d.strftime(f"pinnacle_{league}_%Y-%m-%d_%H%M%S.json")


def _insert_league_suffix(path: str, league: str) -> str:
    """out.json -> out_nba.json (used when PINNACLE_OUTPUT is a fixed file and multiple
    leagues are being scraped in the same invocation, so they don't overwrite each other)."""
    base, ext = os.path.splitext(path)
    ext = ext or ".json"
    return f"{base}_{league}{ext}"


def convert_american_odds(american_odds: list[float]) -> list[float]:
    decimal_odds: list[float] = []
    for odd in american_odds:
        if odd >= 0:
            decimal_odds.append(round(odd / 100 + 1, 3))
        else:
            decimal_odds.append(round(-100 / odd + 1, 3))
    return decimal_odds


def _american_display(x: float | int) -> int | float:
    """API uses American integers (e.g. -115); keep int when whole."""
    xf = float(x)
    if abs(xf - round(xf)) < 1e-6:
        return int(round(xf))
    return xf


def get_pinnacle_odds_from_market_id(
    id_market: int | str,
    all_odds: list[dict[str, Any]],
) -> tuple[list[float], float, list[float]]:
    """
    Returns (decimal_odds, line, american_prices).
    The site UI decimal/american toggle is display-only; Arcadia `price` is American.
    """
    for odds_row in all_odds:
        if not (
            odds_row.get("matchupId") == int(id_market)
            and odds_row.get("type") == "total"
            and odds_row.get("period") == 0
        ):
            continue
        prices = odds_row.get("prices") or []
        if not prices:
            continue
        p0 = prices[0]
        if not all(k in p0 for k in ("price", "points", "participantId")):
            continue
        ordered = sorted(prices, key=lambda x: x["participantId"])
        american_match = [float(x["price"]) for x in ordered]
        decimals = convert_american_odds(american_match)
        return decimals, float(prices[0]["points"]), american_match
    return [], 0.0, []


def _truncate_iso(iso_ts: str) -> str:
    try:
        d = dt.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return iso_ts
    d = d.replace(second=0, microsecond=0)
    return d.isoformat()


def _clean_player_name(description: str) -> str:
    """Return player name only; strip parenthetical notes and trailing 'Total …' wording."""
    player = (description or "").split("(")[0].strip()
    if not player:
        return player
    legacy = re.search(r"(?is)total\s+points\s+by\s+(.+)$", player)
    if legacy:
        return legacy.group(1).strip()
    first_total = re.search(r"(?is)\s+total\s+", player)
    if first_total:
        return player[: first_total.start()].strip()
    return player


def matchup_id_and_slug_from_game_url(raw: str) -> tuple[str, str | None]:
    """Return matchup id string and slug from a full game URL or path fragment."""
    m = _GAME_PATH_RE.search(raw)
    if not m:
        return "", None
    return m.group("mid"), m.group("slug")


def matchup_id_slug_league_from_game_url(raw: str) -> tuple[str, str | None, str | None]:
    """Return (matchup_id, slug, league) from a full game URL or path fragment."""
    m = _GAME_PATH_RE.search(raw)
    if not m:
        return "", None, None
    return m.group("mid"), m.group("slug"), m.group("league").lower()


def _participants_from_slug(slug: str | None) -> list[str]:
    if not slug or "-vs-" not in slug.lower():
        return []
    a, b = re.split(r"-vs-", slug, maxsplit=1, flags=re.IGNORECASE)
    return [p.replace("-", " ").title() for p in (a, b)]


def _normalize_game_url(abs_url: str, expected_league: str) -> str | None:
    mid, slug, league = matchup_id_slug_league_from_game_url(abs_url)
    if not mid or not slug or not league:
        return None
    if league != expected_league:
        return None
    lower = abs_url.lower()
    if "/matchups" in urlparse(lower).path:
        return None
    return urljoin(PINNACLE_ORIGIN, f"/en/basketball/{league}/{slug}/{mid}/#all")


def _worker_scrape_game_batch(
    worker_id: int,
    batch: list[tuple[int, str]],
    league: str,
) -> list[tuple[int, dict[str, Any]]]:
    """
    One Chrome per worker. batch items are (global_index, game_url).
    """
    if not batch:
        return []

    scraper = PinnacleScraper(league)
    driver = scraper._build_driver(worker_id=worker_id, discovery=False)
    out: list[tuple[int, dict[str, Any]]] = []
    n = len(batch)
    try:
        for j, (idx, url) in enumerate(batch, start=1):
            print(f"  [worker {worker_id + 1} {j}/{n}] {url}")
            try:
                game = scraper.scrape_matchup_via_browser(driver, url)
                out.append((idx, game))
            except Exception as exc:
                print(f"    skip (w{worker_id + 1}): {exc}")
                mid, slug = matchup_id_and_slug_from_game_url(url)
                out.append(
                    (
                        idx,
                        {
                            "matchup_id": int(mid) if mid else None,
                            "source_url": url,
                            "slug": slug,
                            "error": str(exc),
                            "props": [],
                            "team_markets": {},
                            "league": league.upper(),
                        },
                    ),
                )
    finally:
        driver.quit()

    return out


class PinnacleScraper:
    """
    Selenium-first player props scraper aligned with repo output layout.
    Works for either "nba" or "wnba" (pass at construction time); each league is
    scraped and saved independently.

    Odds: each prop includes `american_over` / `american_under` (Arcadia native) and
    `decimal_*` (derived). No need to toggle the site — the API always sends American `price`.
    Optional PINNACLE_ODDS_FORMAT=both|decimal|american (default both).
    """

    def __init__(self, league: str, *, multi_league_run: bool = False) -> None:
        league = league.strip().lower()
        if league not in SUPPORTED_LEAGUES:
            raise ValueError(
                f"Unsupported league {league!r}; choose from {SUPPORTED_LEAGUES}",
            )
        self.league = league
        self.matchups_url = LEAGUE_MATCHUPS_URL[league]

        out_file = os.environ.get("PINNACLE_OUTPUT", "").strip()
        out_dir = os.environ.get("PINNACLE_OUTPUT_DIR", "").strip()
        default_dir = os.path.join(_DEFAULT_PINNACLE_BASE_DIR, league)

        if out_file:
            expanded = os.path.expanduser(out_file)
            if out_file.endswith(("/", "\\")) or os.path.isdir(expanded):
                self.output_path = os.path.join(
                    expanded.rstrip("/\\"),
                    _pinnacle_output_filename(league),
                )
            elif expanded.lower().endswith(".json"):
                self.output_path = (
                    _insert_league_suffix(expanded, league)
                    if multi_league_run
                    else expanded
                )
            else:
                self.output_path = os.path.join(expanded, _pinnacle_output_filename(league))
        elif out_dir:
            directory = out_dir
            name = _pinnacle_output_filename(league)
            if directory.endswith(("/", "\\")) or (
                os.path.exists(directory) and os.path.isdir(directory)
            ):
                self.output_path = os.path.join(directory, name)
            elif not os.path.splitext(directory)[1]:
                self.output_path = os.path.join(directory, name)
            else:
                self.output_path = directory
        else:
            self.output_path = os.path.join(default_dir, _pinnacle_output_filename(league))

        fmt = os.environ.get("PINNACLE_ODDS_FORMAT", "both").strip().lower()
        self.odds_format = fmt if fmt in ("both", "decimal", "american") else "both"

        self.page_wait = float(os.environ.get("PINNACLE_PAGE_WAIT", "8"))
        self.max_games = int(os.environ.get("PINNACLE_MAX_GAMES", "50"))

        cpu = os.cpu_count() or 8
        _default_workers = min(6, max(2, cpu // 2))
        try:
            self.parallel_workers = max(
                1,
                int(os.environ.get("PINNACLE_WORKERS", str(_default_workers))),
            )
        except ValueError:
            self.parallel_workers = 1

        self.arcadia_timeout = float(os.environ.get("PINNACLE_ARCADIA_TIMEOUT", "90"))
        self.arcadia_poll = float(os.environ.get("PINNACLE_ARCADIA_POLL", "0.22"))
        self.post_arcadia_buffer = float(
            os.environ.get("PINNACLE_POST_ARCADIA_BUFFER", "0.08"),
        )
        try:
            self.discover_scroll_rounds = int(
                os.environ.get("PINNACLE_DISCOVER_SCROLL_ROUNDS", "8"),
            )
        except ValueError:
            self.discover_scroll_rounds = 8
        self.discover_scroll_pause = float(
            os.environ.get("PINNACLE_DISCOVER_SCROLL_PAUSE", "0.32"),
        )
        try:
            self.discover_retries = max(
                1, int(os.environ.get("PINNACLE_DISCOVER_RETRIES", "3"))
            )
        except ValueError:
            self.discover_retries = 3
        self.discover_retry_pause = float(
            os.environ.get("PINNACLE_DISCOVER_RETRY_PAUSE", "2.5"),
        )
        self.discover_settle = float(os.environ.get("PINNACLE_DISCOVER_SETTLE", "0.65"))
        try:
            self.games_per_worker = max(
                1, int(os.environ.get("PINNACLE_GAMES_PER_WORKER", "2")),
            )
        except ValueError:
            self.games_per_worker = 2
        self._last_effective_parallel: int | None = None

    def _chrome_visible(self) -> bool:
        return os.environ.get("PINNACLE_CHROME_VISIBLE", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def _build_driver(
        self,
        worker_id: int | None = None,
        *,
        discovery: bool = False,
    ):
        if WebDriverWait is None:
            raise RuntimeError("Install selenium (pip install selenium) to run this scraper.")

        from selenium import webdriver  # type: ignore[import-untyped]
        from selenium.webdriver.chrome.options import Options  # type: ignore[import-untyped]

        opts = Options()
        # Discovery must hydrate matchup links; game pages can use eager + optional image block.
        if not discovery:
            if os.environ.get("PINNACLE_MATCHUP_NORMAL_LOAD", "").lower() not in (
                "1",
                "true",
                "yes",
            ):
                opts.page_load_strategy = "eager"
        opts.add_argument("--window-size=1400,1000")
        opts.add_argument("--disable-extensions")
        if not self._chrome_visible():
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--lang=en-US")
        if not discovery and os.environ.get("PINNACLE_BLOCK_IMAGES", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            opts.add_experimental_option(
                "prefs",
                {"profile.managed_default_content_settings.images": 2},
            )

        if worker_id is not None:
            ud = tempfile.mkdtemp(prefix=f"pinnacle_chrome_{self.league}_w{worker_id}_")
            opts.add_argument(f"--user-data-dir={ud}")

        opts.set_capability(
            "goog:loggingPrefs",
            {"performance": "ALL"},
        )

        driver = webdriver.Chrome(options=opts)
        driver.implicitly_wait(0)
        driver.execute_cdp_cmd("Network.enable", {})
        driver.set_page_load_timeout(120)
        return driver

    def _effective_parallel_workers(self, n: int) -> int:
        """Fewer Chromes when each can scrape multiple games (saves cold-start time)."""
        if n <= 0:
            return 1
        per = self.games_per_worker
        ideal = (n + per - 1) // per
        return min(self.parallel_workers, n, max(1, ideal))

    def _scroll_lazy(self, driver, rounds: int | None = None) -> None:
        if By is None:
            return

        r = self.discover_scroll_rounds if rounds is None else rounds
        for _ in range(max(1, r)):
            driver.execute_script(
                "window.scrollBy({top: window.innerHeight * 0.85, behavior: 'instant'})",
            )
            time.sleep(self.discover_scroll_pause)

    def discover_game_urls(self, driver) -> list[str]:
        if WebDriverWait is None:
            raise RuntimeError("Install selenium: pip install selenium")

        cand: set[str] = set()
        for attempt in range(self.discover_retries):
            if attempt > 0:
                print(
                    f"  [{self.league}] No game URLs (discover attempt "
                    f"{attempt + 1}/{self.discover_retries}); "
                    f"waiting {self.discover_retry_pause}s and retrying…",
                )
                time.sleep(self.discover_retry_pause)

            driver.get(self.matchups_url)

            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#root")),
            )
            if self.discover_settle > 0:
                time.sleep(self.discover_settle)
            self._scroll_lazy(driver)

            cand = set()
            anchors = driver.find_elements(By.TAG_NAME, "a")
            for a in anchors:
                raw = (a.get_attribute("href") or "").strip()
                if not raw or "basketball/" not in raw.lower():
                    continue
                canon = _normalize_game_url(raw, self.league)
                if canon:
                    cand.add(canon)

            if not cand:
                for m in _GAME_PATH_RE.finditer(driver.page_source or ""):
                    if m.group("league").lower() != self.league:
                        continue
                    frag = m.group(0)
                    canon = _normalize_game_url(urljoin(PINNACLE_ORIGIN, frag), self.league)
                    if canon:
                        cand.add(canon)

            if cand:
                return sorted(cand)[: self.max_games]

        return []

    def _finished_arcadia_requests(self, driver) -> dict[str, str]:
        """
        Replay Chrome performance logs: maps Network.loadingFinished requestId -> URL for
        guest.api.arcadia requests (browser must have fetched them already).
        """
        urls_by_rid: dict[str, str] = {}
        finished_arcadia: dict[str, str] = {}

        try:
            for entry in driver.get_log("performance"):
                try:
                    msg = json.loads(entry["message"]).get("message") or {}
                except json.JSONDecodeError:
                    continue

                meth = msg.get("method")
                pd = msg.get("params") or {}
                rid = pd.get("requestId")

                if meth == "Network.requestWillBeSent":
                    req = pd.get("request") or {}
                    u = req.get("url") or ""
                    if rid and "guest.api.arcadia.pinnacle.com" in u:
                        urls_by_rid[rid] = u

                elif meth == "Network.responseReceived":
                    resp = pd.get("response") or {}
                    u = resp.get("url") or ""
                    if rid and "guest.api.arcadia.pinnacle.com" in u:
                        urls_by_rid[rid] = u

                elif meth == "Network.loadingFinished" and rid:
                    u = urls_by_rid.get(rid)
                    if (
                        u
                        and "/matchups/" in u
                        and "guest.api.arcadia.pinnacle.com" in u
                    ):
                        finished_arcadia[rid] = u

        except Exception:
            pass

        return finished_arcadia

    def _response_body_json(self, driver, request_id: str) -> Any | None:
        try:
            body = driver.execute_cdp_cmd(
                "Network.getResponseBody",
                {"requestId": request_id},
            )
        except Exception:
            return None

        blob = body.get("body") or ""
        if not blob:
            return None
        if body.get("base64Encoded"):
            try:
                blob = base64.b64decode(blob).decode("utf-8", errors="replace")
            except Exception:
                return None
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None

    def _extract_matchup_arcadia_arrays(
        self,
        driver,
        matchup_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Pull latest straight + related payloads for matchup_id from network log."""

        arcs = self._finished_arcadia_requests(driver)
        best_straight: list[dict[str, Any]] = []
        best_related: list[dict[str, Any]] = []

        slug_rel = f"/matchups/{matchup_id}/related"
        slug_str = f"/matchups/{matchup_id}/markets/related/straight"

        for rid, u in arcs.items():
            parsed = urlparse(u)
            path_norm = parsed.path.replace("//", "/")
            path_ok = f"/matchups/{matchup_id}/" in u
            rel_ok = slug_rel.split("?")[0] in path_norm
            st_ok = slug_str.split("?")[0] in path_norm
            if not path_ok:
                continue
            data_any = self._response_body_json(driver, rid)
            if not isinstance(data_any, list):
                continue

            rows = [
                row for row in data_any if isinstance(row, dict)
            ]
            if rel_ok and rows:
                best_related = rows
            elif st_ok and rows:
                best_straight = rows

        return best_straight, best_related

    def _poll_matchup_arcadia_ready(
        self,
        driver,
        matchup_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Poll performance log until straight + related Arcadia arrays decode, instead of
        long fixed sleeps per attempt.
        """
        deadline = time.time() + max(5.0, self.arcadia_timeout)
        iterations = 0
        while time.time() < deadline:
            straight, related = self._extract_matchup_arcadia_arrays(driver, matchup_id)
            if straight and related:
                if self.post_arcadia_buffer > 0:
                    time.sleep(self.post_arcadia_buffer)
                return straight, related
            time.sleep(max(0.1, self.arcadia_poll))
            iterations += 1
            if iterations % 5 == 0:
                driver.execute_script(
                    "window.dispatchEvent(new Event('resize'));",
                )
        return [], []

    def props_from_arcadia_arrays(
        self,
        straight: list[dict[str, Any]],
        markets_raw: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for market in markets_raw:
            if market.get("type") != "special":
                continue
            unit = market.get("units")
            stat = PLAYER_PROP_UNITS.get(unit)
            if not stat:
                continue
            mid = market.get("id")
            if mid is None:
                continue
            special = market.get("special") or {}
            desc = special.get("description") or ""
            player = _clean_player_name(desc)
            decimals, line, american = get_pinnacle_odds_from_market_id(mid, straight)
            if not decimals or line == 0:
                continue
            row: dict[str, Any] = {
                "stat": stat,
                "player": player,
                "line": line,
                "market_id": mid,
            }
            if self.odds_format in ("both", "decimal"):
                row["decimal_over"] = decimals[0]
                row["decimal_under"] = decimals[1] if len(decimals) > 1 else None
            if self.odds_format in ("both", "american") and american:
                row["american_over"] = _american_display(american[0])
                row["american_under"] = (
                    _american_display(american[1]) if len(american) > 1 else None
                )
            rows.append(row)
        return rows

    def _prices_for_market(
        self,
        market: dict[str, Any],
        straight: list[dict[str, Any]],
        fallback_matchup_id: int | str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find the odds/prices row for a given raw market. Arcadia keys the `straight`
        odds array by `matchupId`, but for per-market special (player prop) rows that
        field actually holds the *market's own id*; for plain team/game markets
        (moneyline/spread/total) it's the real game matchup id. Try the market's own
        id first (mirrors get_pinnacle_odds_from_market_id), then fall back to the
        game's matchup id so this works whichever convention Arcadia used.
        """
        mtype = market.get("type")
        period = market.get("period", 0)
        candidates: list[int] = []
        if market.get("id") is not None:
            candidates.append(int(market["id"]))
        if fallback_matchup_id is not None:
            try:
                fb = int(fallback_matchup_id)
                if fb not in candidates:
                    candidates.append(fb)
            except (TypeError, ValueError):
                pass

        for cand in candidates:
            for row in straight:
                if (
                    row.get("matchupId") == cand
                    and row.get("type") == mtype
                    and row.get("period") == period
                ):
                    prices = row.get("prices") or []
                    if prices:
                        return prices
        return []

    def team_markets_from_arcadia_arrays(
        self,
        straight: list[dict[str, Any]],
        markets_raw: list[dict[str, Any]],
        matchup_id: str,
        participants: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Moneyline / spread / total (game-level, not player props), grouped by type
        then by period (0 = full game, 1/2 = halves, etc. depending on what Pinnacle
        offers for the league)."""
        out: dict[str, list[dict[str, Any]]] = {t: [] for t in TEAM_MARKET_TYPES}

        for market in markets_raw:
            mtype = market.get("type")
            if mtype not in TEAM_MARKET_TYPES:
                continue
            prices = self._prices_for_market(market, straight, matchup_id)
            if not prices:
                continue

            period = market.get("period", 0)
            lines: list[dict[str, Any]] = []
            for p in sorted(prices, key=lambda x: (x.get("participantId") or 0)):
                entry: dict[str, Any] = {}
                pid = p.get("participantId")
                if pid is not None and participants and 0 <= pid < len(participants):
                    entry["team"] = participants[pid]
                elif pid is not None:
                    entry["participant_id"] = pid
                if p.get("designation"):
                    entry["side"] = p["designation"]
                if p.get("points") is not None:
                    entry["points"] = p["points"]
                price = p.get("price")
                if price is not None:
                    entry["american"] = _american_display(price)
                    entry["decimal"] = convert_american_odds([float(price)])[0]
                lines.append(entry)

            if lines:
                out[mtype].append({"period": period, "lines": lines})

        return out

    def scrape_matchup_via_browser(self, driver, game_url: str) -> dict[str, Any]:
        if WebDriverWait is None or By is None or EC is None:
            raise RuntimeError("Install selenium: pip install selenium")

        mid, slug = matchup_id_and_slug_from_game_url(game_url)
        if not mid:
            return {
                "matchup_id": None,
                "source_url": game_url,
                "league": self.league.upper(),
                "start_time": None,
                "label": slug or "",
                "participants": [],
                "props": [],
            }

        driver.get(game_url)
        WebDriverWait(driver, 90).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#root")),
        )
        straight, raw_markets = self._poll_matchup_arcadia_ready(driver, mid)
        if not straight or not raw_markets:
            # Fallback: brief fixed waits for slow CDP / cache (mirrors old behavior)
            for attempt in range(4):
                time.sleep(max(2.0, self.page_wait * 0.5))
                straight, raw_markets = self._extract_matchup_arcadia_arrays(driver, mid)
                if straight and raw_markets:
                    break
                driver.execute_script("window.dispatchEvent(new Event('resize'));")

        props = self.props_from_arcadia_arrays(straight, raw_markets)

        label = slug.replace("-", " ").title().replace(" Vs ", " vs ") if slug else ""
        parsed = urlparse(driver.current_url)
        parts = _participants_from_slug(slug)

        team_markets = self.team_markets_from_arcadia_arrays(straight, raw_markets, mid, parts)

        game: dict[str, Any] = {
            "matchup_id": int(mid),
            "source_url": driver.current_url,
            "path": parsed.path,
            "league": self.league.upper(),
            "start_time": None,
            "label": label,
            "participants": parts,
            "props": props,
            "team_markets": team_markets,
        }

        return game

    def _scrape_games_sequential(self, urls: list[str]) -> list[dict[str, Any]]:
        driver = self._build_driver(discovery=False)
        games_out: list[dict[str, Any]] = []
        try:
            for i, url in enumerate(urls, start=1):
                print(f"  [{self.league} {i}/{len(urls)}] {url}")
                try:
                    games_out.append(self.scrape_matchup_via_browser(driver, url))
                except Exception as exc:
                    print(f"    skip: {exc}")
                    mid, slug = matchup_id_and_slug_from_game_url(url)
                    games_out.append(
                        {
                            "matchup_id": int(mid) if mid else None,
                            "source_url": url,
                            "slug": slug,
                            "error": str(exc),
                            "props": [],
                            "team_markets": {},
                            "league": self.league.upper(),
                        },
                    )
        finally:
            driver.quit()
        return games_out

    def _scrape_games_parallel(self, urls: list[str]) -> list[dict[str, Any]]:
        k = self._effective_parallel_workers(len(urls))
        self._last_effective_parallel = k
        batches: list[list[tuple[int, str]]] = [[] for _ in range(k)]
        for i, url in enumerate(urls):
            batches[i % k].append((i, url))

        merged: list[dict[str, Any] | None] = [None] * len(urls)
        with ThreadPoolExecutor(max_workers=k) as pool:
            futures = {
                pool.submit(_worker_scrape_game_batch, w, batches[w], self.league): w
                for w in range(k)
                if batches[w]
            }
            for fut in as_completed(futures):
                try:
                    for idx, game in fut.result():
                        merged[idx] = game
                except Exception as exc:
                    w = futures[fut]
                    print(f"✗ worker {w + 1} crashed: {exc}")
                    for idx, _url in batches[w]:
                        if merged[idx] is None:
                            merged[idx] = {
                                "matchup_id": None,
                                "source_url": _url,
                                "error": f"worker {w + 1}: {exc}",
                                "props": [],
                                "team_markets": {},
                                "league": self.league.upper(),
                            }

        for i in range(len(urls)):
            if merged[i] is None:
                u = urls[i]
                mid, slug = matchup_id_and_slug_from_game_url(u)
                merged[i] = {
                    "matchup_id": int(mid) if mid else None,
                    "source_url": u,
                    "slug": slug,
                    "error": "internal: no result from worker",
                    "props": [],
                    "team_markets": {},
                    "league": self.league.upper(),
                }

        return list(merged)

    def run(self) -> dict[str, Any]:
        print(
            f"[{self.league}] Using output path: {self.output_path} | "
            f"max_workers={self.parallel_workers} target {self.games_per_worker} games/browser",
        )

        discover_driver = self._build_driver(discovery=True)
        self._last_effective_parallel = None
        try:
            urls = self.discover_game_urls(discover_driver)
        finally:
            discover_driver.quit()

        if not urls:
            print(
                f"[{self.league}] No game URLs parsed from the matchup page; "
                "try PINNACLE_CHROME_VISIBLE=1 or extend PINNACLE_PAGE_WAIT.",
            )
            games_out: list[dict[str, Any]] = []
        elif self.parallel_workers <= 1:
            print(f"[{self.league}] Collected {len(urls)} game URL(s); scraping…")
            games_out = self._scrape_games_sequential(urls)
        else:
            k = self._effective_parallel_workers(len(urls))
            print(
                f"[{self.league}] Collected {len(urls)} game URL(s); scraping with {k} "
                f"worker(s) (≤{self.games_per_worker} games/browser, "
                f"max {self.parallel_workers} configured)…",
            )
            games_out = self._scrape_games_parallel(urls)

        payload: dict[str, Any] = {
            "fetched_at": dt.datetime.now(_PINNACLE_OUTPUT_TZ).isoformat(timespec="seconds"),
            "source": "pinnacle_selenium",
            "sport": "basketball",
            "league": self.league,
            "list_page": self.matchups_url,
            "parallel_workers_configured": self.parallel_workers,
            "parallel_workers_effective": self._last_effective_parallel
            or (1 if urls and self.parallel_workers <= 1 else None),
            "games_per_worker_target": self.games_per_worker,
            "arcadia_timeout_s": self.arcadia_timeout,
            "arcadia_poll_s": self.arcadia_poll,
            "games": games_out,
        }

        out_dir = os.path.dirname(self.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        n_props = sum(len(g.get("props") or []) for g in games_out)
        print(
            f"✓ [{self.league}] Saved {len(games_out)} games, {n_props} player prop rows "
            f"→ {self.output_path}",
        )
        return payload


def _requested_leagues() -> list[str]:
    raw = os.environ.get("PINNACLE_LEAGUES", ",".join(SUPPORTED_LEAGUES))
    leagues = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [x for x in leagues if x not in SUPPORTED_LEAGUES]
    if unknown:
        raise ValueError(
            f"Unknown league(s) in PINNACLE_LEAGUES: {unknown}; "
            f"choose from {SUPPORTED_LEAGUES}",
        )
    # de-dupe, keep order
    seen: set[str] = set()
    ordered = []
    for lg in leagues:
        if lg not in seen:
            seen.add(lg)
            ordered.append(lg)
    return ordered or list(SUPPORTED_LEAGUES)


def run_all() -> dict[str, dict[str, Any]]:
    """Scrape every requested league (default: nba + wnba) and return {league: payload}."""
    leagues = _requested_leagues()
    multi = len(leagues) > 1
    results: dict[str, dict[str, Any]] = {}
    for lg in leagues:
        print(f"=== Starting Pinnacle {lg.upper()} scraper (Selenium)... ===")
        try:
            results[lg] = PinnacleScraper(lg, multi_league_run=multi).run()
        except Exception as e:
            print(f"✗ [{lg}] Error: {e}")
            import traceback

            traceback.print_exc()
    return results


if __name__ == "__main__":
    run_all()