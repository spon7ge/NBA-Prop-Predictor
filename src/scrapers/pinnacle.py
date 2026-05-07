"""
Pinnacle NBA player props scraper via Selenium (no Arcadia REST client).

Flow:
  1) Open https://www.pinnacle.com/en/basketball/nba/matchups/#all
  2) Collect each game URL like …/nba/{slug}/{matchup_id}/#all
  3) Visit each game page; Arcadia responses are decoded from Chrome DevTools
     (responses the site loaded in the browser after navigation).

Requirements: Google Chrome + Selenium-managed chromedriver.
Optional PINNACLE_CHROME_VISIBLE=1 (disable headless if pages load oddly).
PINNACLE_MAX_GAMES caps games per run (default 50). PINNACLE_PAGE_WAIT adjusts post-load pause.
PINNACLE_ODDS_FORMAT=both|decimal|american — Arcadia always sends American `price`; decimal
is derived. The site odds-format switch does not change the API (no need to automate it).

Output: data/props/pinnacle/pinnacle_{YYYY-MM-DD}_{HHMMSS}.json (America/Los_Angeles, PST/PDT). Override with
PINNACLE_OUTPUT (full path to a .json file stays fixed; directory → timestamped name inside) or
PINNACLE_OUTPUT_DIR (directory; timestamped filename appended).

PINNACLE_WORKERS — concurrent Chrome instances for game pages (default 2). Discovery still
uses one short-lived driver; each worker gets its own browser + temp user-data-dir.
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
_DEFAULT_PINNACLE_DIR = os.path.join(_ROOT, "data", "props", "pinnacle")


_PINNACLE_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")


def _pinnacle_output_filename(now: dt.datetime | None = None) -> str:
    """Filesystem-safe name: pinnacle_2026-05-06_213045.json in Pacific time."""
    if now is None:
        d = dt.datetime.now(_PINNACLE_OUTPUT_TZ)
    elif now.tzinfo is None:
        d = now.replace(tzinfo=dt.timezone.utc).astimezone(_PINNACLE_OUTPUT_TZ)
    else:
        d = now.astimezone(_PINNACLE_OUTPUT_TZ)
    return d.strftime("pinnacle_%Y-%m-%d_%H%M%S.json")

NBA_MATCHUPS_URL = "https://www.pinnacle.com/en/basketball/nba/matchups/#all"
PINNACLE_ORIGIN = "https://www.pinnacle.com"

# Related-market `units` -> stable stat key in JSON output (same semantics as Arcadia payloads)
NBA_PLAYER_PROP_UNITS = {
    "Points": "points",
    "Assists": "assists",
    "Rebounds": "rebounds",
    "PointsReboundsAssist": "points_rebounds_assists",
}

_GAME_PATH_RE = re.compile(
    r"/en/basketball/nba/(?P<slug>[\w.-]+)/(?P<mid>\d+)/?",
    re.IGNORECASE,
)


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
    player = description.split("(")[0].strip()
    if "Total Points by" in player:
        player = player.split("Total Points by")[1].strip()
    return player


def matchup_id_and_slug_from_game_url(raw: str) -> tuple[str, str | None]:
    """Return matchup id string and slug from a full game URL or path fragment."""
    m = _GAME_PATH_RE.search(raw)
    if not m:
        return "", None
    return m.group("mid"), m.group("slug")


def _participants_from_slug(slug: str | None) -> list[str]:
    if not slug or "-vs-" not in slug.lower():
        return []
    a, b = re.split(r"-vs-", slug, maxsplit=1, flags=re.IGNORECASE)
    return [p.replace("-", " ").title() for p in (a, b)]


def _normalize_game_url(abs_url: str) -> str | None:
    mid, slug = matchup_id_and_slug_from_game_url(abs_url)
    if not mid or not slug:
        return None
    lower = abs_url.lower()
    if "/matchups" in urlparse(lower).path:
        return None
    return urljoin(PINNACLE_ORIGIN, f"/en/basketball/nba/{slug}/{mid}/#all")


def _worker_scrape_game_batch(
    worker_id: int,
    batch: list[tuple[int, str]],
) -> list[tuple[int, dict[str, Any]]]:
    """
    One Chrome per worker. batch items are (global_index, game_url).
    """
    if not batch:
        return []

    scraper = PinnacleNBAScraper()
    driver = scraper._build_driver(worker_id=worker_id)
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
                            "league": "NBA",
                        },
                    ),
                )
    finally:
        driver.quit()

    return out


class PinnacleNBAScraper:
    """
    Selenium-first NBA matchup player props scraper aligned with repo output layout.

    Odds: each prop includes `american_over` / `american_under` (Arcadia native) and
    `decimal_*` (derived). No need to toggle the site — the API always sends American `price`.
    Optional PINNACLE_ODDS_FORMAT=both|decimal|american (default both).
    """

    def __init__(self) -> None:
        out_file = os.environ.get("PINNACLE_OUTPUT", "").strip()
        out_dir = os.environ.get("PINNACLE_OUTPUT_DIR", "").strip()

        if out_file:
            expanded = os.path.expanduser(out_file)
            if out_file.endswith(("/", "\\")) or os.path.isdir(expanded):
                self.output_path = os.path.join(
                    expanded.rstrip("/\\"),
                    _pinnacle_output_filename(),
                )
            elif expanded.lower().endswith(".json"):
                self.output_path = expanded
            else:
                self.output_path = os.path.join(expanded, _pinnacle_output_filename())
        elif out_dir:
            directory = out_dir
            name = _pinnacle_output_filename()
            if directory.endswith(("/", "\\")) or (
                os.path.exists(directory) and os.path.isdir(directory)
            ):
                self.output_path = os.path.join(directory, name)
            elif not os.path.splitext(directory)[1]:
                self.output_path = os.path.join(directory, name)
            else:
                self.output_path = directory
        else:
            self.output_path = os.path.join(_DEFAULT_PINNACLE_DIR, _pinnacle_output_filename())

        fmt = os.environ.get("PINNACLE_ODDS_FORMAT", "both").strip().lower()
        self.odds_format = fmt if fmt in ("both", "decimal", "american") else "both"

        self.page_wait = float(os.environ.get("PINNACLE_PAGE_WAIT", "14"))
        self.max_games = int(os.environ.get("PINNACLE_MAX_GAMES", "50"))
        try:
            self.parallel_workers = max(1, int(os.environ.get("PINNACLE_WORKERS", "2")))
        except ValueError:
            self.parallel_workers = 1

    def _chrome_visible(self) -> bool:
        return os.environ.get("PINNACLE_CHROME_VISIBLE", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def _build_driver(self, worker_id: int | None = None):
        if WebDriverWait is None:
            raise RuntimeError("Install selenium (pip install selenium) to run this scraper.")

        from selenium import webdriver  # type: ignore[import-untyped]
        from selenium.webdriver.chrome.options import Options  # type: ignore[import-untyped]

        opts = Options()
        opts.add_argument("--window-size=1400,1000")
        opts.add_argument("--disable-extensions")
        if not self._chrome_visible():
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--lang=en-US")

        if worker_id is not None:
            ud = tempfile.mkdtemp(prefix=f"pinnacle_chrome_w{worker_id}_")
            opts.add_argument(f"--user-data-dir={ud}")

        opts.set_capability(
            "goog:loggingPrefs",
            {"performance": "ALL"},
        )

        driver = webdriver.Chrome(options=opts)
        driver.implicitly_wait(2)
        driver.execute_cdp_cmd("Network.enable", {})
        driver.set_page_load_timeout(120)
        return driver

    def _scroll_lazy(self, driver, rounds: int = 12) -> None:
        if By is None:
            return

        for _ in range(rounds):
            driver.execute_script(
                "window.scrollBy({top: window.innerHeight * 0.85, behavior: 'instant'})",
            )
            time.sleep(0.45)

    def discover_game_urls(self, driver) -> list[str]:
        if WebDriverWait is None:
            raise RuntimeError("Install selenium: pip install selenium")

        driver.get(NBA_MATCHUPS_URL)

        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#root")),
        )
        self._scroll_lazy(driver)

        cand: set[str] = set()

        anchors = driver.find_elements(By.TAG_NAME, "a")
        for a in anchors:
            raw = (a.get_attribute("href") or "").strip()
            if not raw or "basketball/nba/" not in raw.lower():
                continue
            canon = _normalize_game_url(raw)
            if canon:
                cand.add(canon)

        if not cand:
            for m in _GAME_PATH_RE.finditer(driver.page_source or ""):
                frag = m.group(0)
                canon = _normalize_game_url(urljoin(PINNACLE_ORIGIN, frag))
                if canon:
                    cand.add(canon)

        return sorted(cand)[: self.max_games]

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
            stat = NBA_PLAYER_PROP_UNITS.get(unit)
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

    def scrape_matchup_via_browser(self, driver, game_url: str) -> dict[str, Any]:
        if WebDriverWait is None or By is None or EC is None:
            raise RuntimeError("Install selenium: pip install selenium")

        mid, slug = matchup_id_and_slug_from_game_url(game_url)
        if not mid:
            return {
                "matchup_id": None,
                "source_url": game_url,
                "league": "NBA",
                "start_time": None,
                "label": slug or "",
                "participants": [],
                "props": [],
            }

        driver.get(game_url)
        WebDriverWait(driver, 90).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#root")),
        )
        straight, raw_markets = [], []
        for attempt in range(6):
            time.sleep(max(4.0, self.page_wait) if attempt == 0 else 3.5)
            straight, raw_markets = self._extract_matchup_arcadia_arrays(driver, mid)
            if straight and raw_markets:
                break
            driver.execute_script("window.dispatchEvent(new Event('resize'));")

        props = self.props_from_arcadia_arrays(straight, raw_markets)

        label = slug.replace("-", " ").title().replace(" Vs ", " vs ") if slug else ""
        parsed = urlparse(driver.current_url)
        parts = _participants_from_slug(slug)

        game: dict[str, Any] = {
            "matchup_id": int(mid),
            "source_url": driver.current_url,
            "path": parsed.path,
            "league": "NBA",
            "start_time": None,
            "label": label,
            "participants": parts,
            "props": props,
        }

        return game

    def _scrape_games_sequential(self, urls: list[str]) -> list[dict[str, Any]]:
        driver = self._build_driver()
        games_out: list[dict[str, Any]] = []
        try:
            for i, url in enumerate(urls, start=1):
                print(f"  [{i}/{len(urls)}] {url}")
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
                            "league": "NBA",
                        },
                    )
        finally:
            driver.quit()
        return games_out

    def _scrape_games_parallel(self, urls: list[str]) -> list[dict[str, Any]]:
        k = min(self.parallel_workers, max(1, len(urls)))
        batches: list[list[tuple[int, str]]] = [[] for _ in range(k)]
        for i, url in enumerate(urls):
            batches[i % k].append((i, url))

        merged: list[dict[str, Any] | None] = [None] * len(urls)
        with ThreadPoolExecutor(max_workers=k) as pool:
            futures = {
                pool.submit(_worker_scrape_game_batch, w, batches[w]): w
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
                                "league": "NBA",
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
                    "league": "NBA",
                }

        return list(merged)

    def run(self) -> dict[str, Any]:
        print(f"Using output path: {self.output_path}")
        if self.parallel_workers > 1:
            print(f"Parallel workers: {self.parallel_workers}")

        discover_driver = self._build_driver()
        try:
            urls = self.discover_game_urls(discover_driver)
        finally:
            discover_driver.quit()

        if not urls:
            print(
                "No game URLs parsed from the matchup page; "
                "try PINNACLE_CHROME_VISIBLE=1 or extend PINNACLE_PAGE_WAIT.",
            )
            games_out: list[dict[str, Any]] = []
        elif self.parallel_workers <= 1:
            print(f"Collected {len(urls)} game URL(s); scraping…")
            games_out = self._scrape_games_sequential(urls)
        else:
            k = min(self.parallel_workers, len(urls))
            print(f"Collected {len(urls)} game URL(s); scraping with {k} workers…")
            games_out = self._scrape_games_parallel(urls)

        payload: dict[str, Any] = {
            "fetched_at": dt.datetime.now(_PINNACLE_OUTPUT_TZ).isoformat(timespec="seconds"),
            "source": "pinnacle_selenium",
            "sport": "nba",
            "list_page": NBA_MATCHUPS_URL,
            "parallel_workers": self.parallel_workers,
            "games": games_out,
        }

        out_dir = os.path.dirname(self.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        n_props = sum(len(g.get("props") or []) for g in games_out)
        print(f"✓ Saved {len(games_out)} games, {n_props} player prop rows → {self.output_path}")
        return payload


if __name__ == "__main__":
    print("Starting Pinnacle NBA scraper (Selenium)...")
    try:
        PinnacleNBAScraper().run()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
