"""
DraftKings Pick 6 NBA Prop Lines Scraper (Playwright)
------------------------------------------------------
Uses a real browser to bypass Akamai bot detection.
Intercepts the network response containing player lines and reads Pick6 DOM cards.

Setup:
    pip install playwright
    playwright install chromium

Output: ``data/props/draftking6/draftkings6_{YYYY-MM-DD}_{HHMMSS}.json`` (America/Los_Angeles).
Override with ``DRAFTKINGS6_OUTPUT`` (full ``.json`` path, or a directory → timestamped name inside) or
``DRAFTKINGS6_OUTPUT_DIR``. CLI ``-o`` wins over env when provided.

Raw XHR/debug dump: default ``data/props/draftking6/debug/draftkings6_raw_{timestamp}.json``.
Set ``DRAFTKINGS6_RAW_OUTPUT`` to a full path, or ``DRAFTKINGS6_NO_RAW=1`` to skip.

Env knobs (optional):
    DRAFTKINGS6_WORKERS — parallel carousel shards for NBA (default 4; same idea as PINNACLE_WORKERS).

Usage:
    python draftkings6.py                   # NBA slate; 4 parallel carousel workers by default
    python draftkings6.py -w 1             # fully sequential (single context, all tabs in order)
    python draftkings6.py -w 6             # up to 6 parallel shards (capped by tab count)
    python draftkings6.py -o props6.json
    python draftkings6.py --headful         # watch the browser (debug); forces -w 1
    python draftkings6.py --url "https://..."  # paste exact board URL from Pick6
    python draftkings6.py --sport NFL       # lands on /?sport=NFL
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from playwright.async_api import Browser, async_playwright, Page, Response

_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment,misc]

if load_dotenv:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))

_DEFAULT_DK_PROPS_DIR = os.path.join(_REPO_ROOT, "data", "props", "draftking6")
_DEFAULT_DK_DEBUG_DIR = os.path.join(_DEFAULT_DK_PROPS_DIR, "debug")

PICK6_URL = "https://pick6.draftkings.com"

# NBA props board (/category/{id}?sport=NBA&pickGroup=...). pickGroup changes when DK opens a new slate.
DEFAULT_NBA_CATEGORY_ID = "16"
DEFAULT_NBA_PICK_GROUP  = "147290"

# JSON field hints — responses are wrapped various ways; see _walk_find_line_list
LINES_KEY_HINTS = [
    "statType", "playerName", "lineValue", "line", "projection",
    "targetValue", "pickableId", "pickSixMarketId",
]

# Longer phrases first (avoid matching "Points" inside "1Q Points")
_STAT_FROM_CARD = [
    "1Q Points", "1st Point", "Superstars",
    "3-Pointers", "Quick Hits", "Combos",
    "Rebounds", "Assists", "Steals", "Blocks", "Turnovers", "PRA", "Points",
]

# Carousel labels on NBA category boards (exact match helps Playwright clicks)
NBA_STAT_MARKET_PILLS = [
    "Points",
    "Superstars",
    "3-Pointers",
    "Quick Hits",
    "Combos",
    "Rebounds",
    "Assists",
]

try:
    _DEFAULT_PAR_WORKERS = max(1, int(os.environ.get("DRAFTKINGS6_WORKERS", "4")))
except ValueError:
    _DEFAULT_PAR_WORKERS = 4


def _draftkings6_props_filename(now: datetime | None = None) -> str:
    if now is None:
        d = datetime.now(_OUTPUT_TZ)
    elif now.tzinfo is None:
        d = now.replace(tzinfo=timezone.utc).astimezone(_OUTPUT_TZ)
    else:
        d = now.astimezone(_OUTPUT_TZ)
    return d.strftime("draftkings6_%Y-%m-%d_%H%M%S.json")


def _draftkings6_raw_filename(now: datetime | None = None) -> str:
    if now is None:
        d = datetime.now(_OUTPUT_TZ)
    elif now.tzinfo is None:
        d = now.replace(tzinfo=timezone.utc).astimezone(_OUTPUT_TZ)
    else:
        d = now.astimezone(_OUTPUT_TZ)
    return d.strftime("draftkings6_raw_%Y-%m-%d_%H%M%S.json")


def resolve_props_output_path(cli_output: str | None) -> str:
    """CLI ``-o`` > ``DRAFTKINGS6_OUTPUT`` > ``DRAFTKINGS6_OUTPUT_DIR`` > ``data/props/draftking6``."""
    if cli_output and str(cli_output).strip():
        return os.path.expanduser(str(cli_output).strip())

    out_file = os.environ.get("DRAFTKINGS6_OUTPUT", "").strip()
    out_dir = os.environ.get("DRAFTKINGS6_OUTPUT_DIR", "").strip()

    if out_file:
        expanded = os.path.expanduser(out_file)
        if out_file.endswith(("/", "\\")) or os.path.isdir(expanded):
            return os.path.join(
                expanded.rstrip("/\\"),
                _draftkings6_props_filename(),
            )
        if expanded.lower().endswith(".json"):
            return expanded
        return os.path.join(expanded, _draftkings6_props_filename())

    if out_dir:
        directory = os.path.expanduser(out_dir)
        name = _draftkings6_props_filename()
        if directory.endswith(("/", "\\")) or (
            os.path.exists(directory) and os.path.isdir(directory)
        ):
            return os.path.join(directory.rstrip("/\\"), name)
        if not os.path.splitext(directory)[1]:
            return os.path.join(directory, name)
        return directory

    os.makedirs(_DEFAULT_DK_PROPS_DIR, exist_ok=True)
    return os.path.join(_DEFAULT_DK_PROPS_DIR, _draftkings6_props_filename())


def resolve_raw_responses_path() -> str | None:
    if os.environ.get("DRAFTKINGS6_NO_RAW", "").lower() in ("1", "true", "yes"):
        return None
    custom = os.environ.get("DRAFTKINGS6_RAW_OUTPUT", "").strip()
    if custom:
        exp = os.path.expanduser(custom)
        if exp.lower().endswith(".json"):
            return exp
        if exp.endswith(("/", "\\")) or (os.path.exists(exp) and os.path.isdir(exp)):
            return os.path.join(exp.rstrip("/\\"), _draftkings6_raw_filename())
        return os.path.join(exp, _draftkings6_raw_filename())
    os.makedirs(_DEFAULT_DK_DEBUG_DIR, exist_ok=True)
    return os.path.join(_DEFAULT_DK_DEBUG_DIR, _draftkings6_raw_filename())


PLAYWRIGHT_CONTEXT_KWARGS = {
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36"
    ),
    "locale": "en-US",
    "viewport": {"width": 1280, "height": 900},
}


def _walk_find_line_list(obj, depth: int = 0):
    """Depth-first search for a list of dicts that look like Pick 6 lines."""
    if depth > 15:
        return None
    if isinstance(obj, list) and obj:
        if isinstance(obj[0], dict):
            keys = obj[0].keys()
            if any(h in keys for h in LINES_KEY_HINTS):
                return obj
        for item in obj:
            found = _walk_find_line_list(item, depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, dict):
        for v in obj.values():
            found = _walk_find_line_list(v, depth + 1)
            if found is not None:
                return found
    return None


def looks_like_lines(body) -> bool:
    return _walk_find_line_list(body) is not None


def extract_lines(body) -> list[dict]:
    found = _walk_find_line_list(body)
    return list(found) if found else []


def _stat_from_card_text(text: str) -> str:
    for kw in _STAT_FROM_CARD:
        if kw in text:
            return kw
    return ""


def _line_from_card_text(text: str, multiplier: float | None) -> float | None:
    """Pick the main prop line (e.g. 29.5), not the '1x' multiplier."""
    candidates: list[tuple[int, float]] = []
    for m in re.finditer(r"\b(\d+\.\d+)\b", text):
        v = float(m.group(1))
        if 0.5 <= v <= 80:
            candidates.append((m.start(), v))
    for _, v in sorted(candidates, key=lambda x: x[0]):
        if multiplier is not None and abs(v - multiplier) < 0.001:
            continue
        # Skip tiny decimals that are usually UI multipliers mis-read
        if v < 1.25:
            continue
        return v
    return None


def _dom_card_rows_to_raw(rows: list, category_pill: str = "") -> list[dict]:
    """category_pill: active carousel label — used when card copy omits stat name (e.g. Superstars)."""
    out: list[dict] = []
    fallback = (category_pill or "").strip()

    for r in rows:
        text = (r.get("text") or "").strip()
        mults = [float(x) for x in re.findall(r"(\d+\.?\d*)\s*x\b", text, re.I)]
        mult = mults[-1] if mults else None
        line = _line_from_card_text(text, mult)
        stat = _stat_from_card_text(text) or fallback
        pid = str(r.get("pickableId") or "")
        ply = (r.get("player") or "").strip()
        if not pid and not ply:
            continue
        out.append({
            "playerName": ply,
            "sportAbbreviation": "NBA",
            "statType": stat,
            "lineValue": line,
            "payoutMultiplier": mult,
            "id": pid,
            "pick6MarketTab": fallback,
        })
    return out


async def extract_lines_from_dom(page: Page, *, category_pill: str = "") -> list[dict]:
    """
    Pick6 hydrates picks in the DOM; the SSR JSON stream rarely appears as fetch/XHR JSON.
    Cards expose stable hooks: data-testid=\"playerStatCard\", data-pickable-id.
    """
    rows = await page.evaluate(
        """() => {
          const cards = document.querySelectorAll('[data-testid="playerStatCard"]');
          return Array.from(cards).map((el) => {
            const pickableId = el.getAttribute('data-pickable-id') || '';
            const btn = el.querySelector('[data-testid="cardButton"]');
            const aria = (btn && btn.getAttribute('aria-label')) || '';
            let player = '';
            const am = /^Open (.+?)['\\u2019\\u02BC]s stat details$/i.exec(aria);
            if (am) player = am[1].trim();
            return { pickableId, player, text: (el.innerText || '').trim() };
          });
        }"""
    )
    return _dom_card_rows_to_raw(rows, category_pill=category_pill)


async def _click_pick6_market_pill(page: Page, label: str) -> bool:
    """Activate a stat carousel chip (horizontal list under the sport)."""
    patterns = [
        lambda: page.get_by_role("button", name=label, exact=True),
        lambda: page.get_by_role("tab", name=label, exact=True),
        lambda: page.get_by_role("button", name=re.compile("^" + re.escape(label) + "$", re.I)),
    ]
    for get_loc in patterns:
        loc = get_loc()
        try:
            if await loc.count() == 0:
                continue
            target = loc.first
            await target.scroll_into_view_if_needed(timeout=10_000)
            await target.click(timeout=10_000)
            return True
        except Exception:
            continue
    return False


async def _scroll_to_load_lazy_cards(page: Page) -> None:
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(400)
    for _ in range(5):
        await page.evaluate("window.scrollBy(0, Math.min(700, window.innerHeight))")
        await page.wait_for_timeout(450)
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(600)


async def scrape_dom_all_nba_market_pills(page: Page) -> list[dict]:
    """
    Each pill replaces the pick list. Collect unique pickable ids across tabs.
    """
    collected: list[dict] = []
    seen: set[str] = set()

    for label in NBA_STAT_MARKET_PILLS:
        ok = await _click_pick6_market_pill(page, label)
        if not ok:
            print(f"  [skip] Could not activate market pill \"{label}\"")
            continue
        await page.wait_for_timeout(1_800)
        await _scroll_to_load_lazy_cards(page)
        chunk = await extract_lines_from_dom(page, category_pill=label)
        new_n = 0
        for row in chunk:
            pid = str(row.get("id") or "")
            if pid and pid in seen:
                continue
            if pid:
                seen.add(pid)
            collected.append(row)
            new_n += 1
        print(f"  [ok] \"{label}\": +{new_n} picks (running total {len(collected)})")

    return collected


def _nba_pill_batches(workers: int) -> list[list[str]]:
    """Split carousel labels across parallel workers (Pinnacle-style sharding)."""
    pills = NBA_STAT_MARKET_PILLS
    w = max(1, min(workers, len(pills)))
    buckets: list[list[str]] = [[] for _ in range(w)]
    for i, lbl in enumerate(pills):
        buckets[i % w].append(lbl)
    return [b for b in buckets if b]


def _attach_response_logger(page: Page, raw_dump: list, json_lines_out: list) -> None:
    async def handle_response(response: Response):
        url = response.url
        if any(ext in url for ext in [".js", ".css", ".png", ".svg", ".woff"]):
            return
        ct = (response.headers.get("content-type") or "").lower()
        body = None
        try:
            if "json" in ct:
                body = await response.json()
            elif "pick6.draftkings.com" in url or "api.draftkings.com" in url:
                txt = (await response.text()).strip()
                if txt.startswith("{") or txt.startswith("["):
                    body = json.loads(txt)
        except Exception:
            return
        if body is None:
            return
        raw_dump.append({"url": url, "data": body})
        if looks_like_lines(body):
            lines = extract_lines(body)
            if lines:
                json_lines_out.extend(lines)

    page.on("response", handle_response)


async def _prepare_pick6_listing(page: Page, sport: str, listing_url: str) -> None:
    await page.goto(listing_url, wait_until="domcontentloaded", timeout=90_000)
    await page.wait_for_timeout(5_000)
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(2_000)

    for tab_text in [sport, "NBA", "Basketball"]:
        try:
            btn = page.get_by_role("button", name=tab_text).first
            if await btn.is_visible():
                await btn.click()
                await page.wait_for_timeout(2_000)
                break
        except Exception:
            pass

    try:
        await page.wait_for_selector(
            '[data-testid="playerStatCard"]',
            timeout=45_000,
        )
    except Exception:
        pass


async def _nba_parallel_worker_batch(
    browser: Browser,
    worker_id: int,
    pill_batch: list[str],
    *,
    listing_url: str,
    sport: str,
) -> tuple[list[dict], list, list]:
    """
    One browser context, one navigation, then sequentially activate each pill in this shard.
    """
    raw_dump: list = []
    json_lines: list = []
    dom_out: list[dict] = []
    seen: set[str] = set()

    context = await browser.new_context(**PLAYWRIGHT_CONTEXT_KWARGS)
    page = await context.new_page()
    try:
        _attach_response_logger(page, raw_dump, json_lines)
        print(f"  [worker {worker_id + 1}] loading board ({len(pill_batch)} carousel tab(s))...")
        await _prepare_pick6_listing(page, sport, listing_url)

        for label in pill_batch:
            ok = await _click_pick6_market_pill(page, label)
            if not ok:
                print(f"  [worker {worker_id + 1} skip] pill \"{label}\"")
                continue
            await page.wait_for_timeout(1_800)
            await _scroll_to_load_lazy_cards(page)
            chunk = await extract_lines_from_dom(page, category_pill=label)
            new_n = 0
            for row in chunk:
                pid = str(row.get("id") or "")
                if pid and pid in seen:
                    continue
                if pid:
                    seen.add(pid)
                dom_out.append(row)
                new_n += 1
            print(
                f"  [worker {worker_id + 1}] \"{label}\": +{new_n} "
                f"(shard total {len(dom_out)})"
            )
        return dom_out, raw_dump, json_lines
    finally:
        await context.close()


async def scrape_nba_markets_parallel(
    browser: Browser,
    *,
    listing_url: str,
    sport: str,
    parallel_workers: int,
) -> tuple[list[dict], list, list]:
    """
    Run carousel shards concurrently (similar idea to PINNACLE_WORKERS — each shard = one tab chain).
    """
    batches = _nba_pill_batches(parallel_workers)
    tasks = [
        _nba_parallel_worker_batch(
            browser,
            wid,
            batch,
            listing_url=listing_url,
            sport=sport,
        )
        for wid, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks)
    merged_dom: list[dict] = []
    merged_raw: list = []
    merged_json_lines: list = []
    id_seen: set[str] = set()
    for dom_part, raw_part, json_part in results:
        merged_raw.extend(raw_part)
        merged_json_lines.extend(json_part)
        for row in dom_part:
            pid = str(row.get("id") or "")
            if pid and pid in id_seen:
                continue
            if pid:
                id_seen.add(pid)
            merged_dom.append(row)
    return merged_dom, merged_raw, merged_json_lines


def build_start_url(
    sport: str,
    *,
    full_url: str | None = None,
    category_id: str = DEFAULT_NBA_CATEGORY_ID,
    pick_group: str = DEFAULT_NBA_PICK_GROUP,
) -> str:
    """
    NBA defaults to the category props page (see Pick6 UI).
    Other sports use the generic home query until a category URL is known.
    """
    if full_url:
        return full_url.strip()
    s = (sport or "NBA").upper()
    if s == "NBA":
        return f"{PICK6_URL}/category/{category_id}?sport=NBA&pickGroup={pick_group}"
    return f"{PICK6_URL}/?sport={sport}"


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse(raw_lines: list[dict], sport_filter: str = "NBA") -> list[dict]:
    """
    Pick 6 uses fixed payout multipliers, not traditional odds.
    A pick pays out at a multiplier (e.g. 3x) based on how many legs
    you parlay — no per-selection over/under odds exist.
    """
    props = []
    for entry in raw_lines:
        sport = (
            entry.get("sport") or entry.get("league") or
            entry.get("sportAbbreviation") or ""
        ).upper()
        if sport_filter and sport_filter.upper() not in sport:
            continue

        multiplier = (
            entry.get("payoutMultiplier")
            or entry.get("multiplier")
            or entry.get("boost")
            or entry.get("payout")
            or entry.get("overPayout")
        )
        try:
            multiplier = float(multiplier) if multiplier is not None else None
        except (ValueError, TypeError):
            multiplier = None

        props.append({
            "player":            (entry.get("playerName")       or entry.get("displayName")   or entry.get("name", "")),
            "team":              (entry.get("teamAbbreviation") or entry.get("teamShortName") or entry.get("team", "")),
            "position":           entry.get("position", ""),
            "prop_type":         (entry.get("statType")         or entry.get("stat_type")     or entry.get("category", "")),
            "line":              (entry.get("lineValue")        or entry.get("line")          or entry.get("value")),
            "payout_multiplier":  multiplier,
            "opponent":          (entry.get("opponentAbbreviation") or entry.get("opponent", "")),
            "game_time":         (entry.get("startTime")        or entry.get("gameStartTime") or entry.get("game_time", "")),
            "status":             entry.get("status", ""),
            "prop_id":            str(entry.get("id") or entry.get("pickId") or ""),
            "pick6_market_tab":  (entry.get("pick6MarketTab") or ""),
        })
    return props

# ── Browser scraping ──────────────────────────────────────────────────────────

async def scrape(
    sport: str = "NBA",
    headful: bool = False,
    *,
    start_url: str | None = None,
    category_id: str = DEFAULT_NBA_CATEGORY_ID,
    pick_group: str = DEFAULT_NBA_PICK_GROUP,
    parallel_workers: int = _DEFAULT_PAR_WORKERS,
) -> list[dict]:
    all_raw:  list = []
    raw_dump: list[dict] = []

    listing_url = build_start_url(
        sport,
        full_url=start_url,
        category_id=category_id,
        pick_group=pick_group,
    )

    sport_u = (sport or "").strip().upper()
    pw = max(1, parallel_workers)
    if headful and pw > 1:
        print("  [warn] --headful: parallel disabled (single worker)")
        pw = 1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headful)
        use_parallel_nba = sport_u == "NBA" and pw > 1

        if use_parallel_nba:
            n_shards = min(pw, len(NBA_STAT_MARKET_PILLS))
            print(f"NBA board: {n_shards} parallel worker(s) (carousel tabs split like Pinnacle URL sharding)")
            dom_lines, extra_raw, extra_json_lines = await scrape_nba_markets_parallel(
                browser,
                listing_url=listing_url,
                sport=sport,
                parallel_workers=pw,
            )
            raw_dump.extend(extra_raw)
            all_raw.extend(extra_json_lines)
            all_raw.extend(dom_lines)
        else:
            context = await browser.new_context(**PLAYWRIGHT_CONTEXT_KWARGS)
            page = await context.new_page()
            _attach_response_logger(page, raw_dump, all_raw)

            print(f"Navigating to {listing_url} ...")
            await _prepare_pick6_listing(page, sport, listing_url)

            dom_lines_single: list[dict] = []
            if sport_u == "NBA":
                dom_lines_single = await scrape_dom_all_nba_market_pills(page)
            else:
                await _scroll_to_load_lazy_cards(page)
                chunk = await extract_lines_from_dom(page)
                if chunk:
                    print(f"  [ok] Parsed {len(chunk)} lines from DOM (playerStatCard)")
                dom_lines_single = chunk
            if dom_lines_single:
                all_raw.extend(dom_lines_single)
            await context.close()

        await browser.close()

    raw_path = resolve_raw_responses_path()
    if raw_path:
        parent = os.path.dirname(os.path.abspath(raw_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_dump, f, indent=2)
        print(f"Saved {len(raw_dump)} intercepted responses -> {raw_path}")

    return all_raw


# ── Output ────────────────────────────────────────────────────────────────────

def save(props: list[dict], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    out = {
        "source":     "DraftKings Pick 6",
        "league":     "NBA",
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count":      len(props),
        "props":      props,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(props)} props -> {path}")


class DraftKingsPick6Scraper:
    """Same role as ``PinnacleNBAScraper``: env-resolved output dir, single ``run()`` entry."""

    def __init__(
        self,
        *,
        sport: str = "NBA",
        headful: bool = False,
        start_url: str | None = None,
        category_id: str = DEFAULT_NBA_CATEGORY_ID,
        pick_group: str = DEFAULT_NBA_PICK_GROUP,
        parallel_workers: int | None = None,
        output_path: str | None = None,
        list_types: bool = False,
    ) -> None:
        self.sport = sport
        self.headful = headful
        self.start_url = start_url
        self.category_id = category_id
        self.pick_group = pick_group
        self.parallel_workers = (
            parallel_workers if parallel_workers is not None else _DEFAULT_PAR_WORKERS
        )
        self.output_path = output_path
        self.list_types = list_types

    async def run_async(self) -> str | None:
        raw_lines = await scrape(
            sport=self.sport,
            headful=self.headful,
            start_url=self.start_url,
            category_id=self.category_id,
            pick_group=self.pick_group,
            parallel_workers=self.parallel_workers,
        )
        if not raw_lines:
            print(
                "\nNo props found (network + DOM).\n"
                "  1. Run with headful=True — confirm the board loads (login / geo / bot wall).\n"
                "  2. Check the raw debug JSON under data/props/draftking6/debug/.\n"
                "  3. Update start_url / pick_group if the slate changed."
            )
            return None
        props = parse(raw_lines, sport_filter=self.sport)
        seen: set[str] = set()
        unique: list[dict] = []
        for p in props:
            uid = p["prop_id"] or f"{p['player']}_{p['prop_type']}"
            if uid not in seen:
                seen.add(uid)
                unique.append(p)
        print(f"Parsed {len(unique)} unique props")
        if self.list_types:
            types = sorted({p["prop_type"] for p in unique if p["prop_type"]})
            print(f"\nStat types ({len(types)}):")
            for t in types:
                print(f"  * {t}")
            return None
        out_path = resolve_props_output_path(self.output_path)
        save(unique, out_path)
        return out_path

    def run(self) -> str | None:
        return asyncio.run(self.run_async())


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main_async(args):
    scraper = DraftKingsPick6Scraper(
        sport=args.sport,
        headful=args.headful,
        start_url=args.url,
        category_id=args.category,
        pick_group=args.pick_group,
        parallel_workers=args.workers,
        output_path=args.output,
        list_types=args.list_types,
    )
    await scraper.run_async()


def main():
    parser = argparse.ArgumentParser(description="DraftKings Pick 6 props -> JSON (Playwright)")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output JSON path (overrides DRAFTKINGS6_OUTPUT / DRAFTKINGS6_OUTPUT_DIR; "
            "default: data/props/draftking6/draftkings6_<ts>.json)"
        ),
    )
    parser.add_argument("--sport",      "-s", default="NBA",       help="Sport (default: NBA)")
    parser.add_argument(
        "--url",
        "-u",
        default=None,
        help="Full Pick6 board URL (overrides category / pick-group for NBA)",
    )
    parser.add_argument(
        "--category",
        "-c",
        default=DEFAULT_NBA_CATEGORY_ID,
        help=f"NBA /category/{{id}}/ path segment (default: {DEFAULT_NBA_CATEGORY_ID})",
    )
    parser.add_argument(
        "--pick-group",
        "-g",
        default=DEFAULT_NBA_PICK_GROUP,
        help=f"NBA pickGroup= query param (default: {DEFAULT_NBA_PICK_GROUP}; rotates with new slates)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=_DEFAULT_PAR_WORKERS,
        help=(
            "NBA only: parallel carousel shards (DRAFTKINGS6_WORKERS env default). "
            "Use 1 for fully sequential. Ignored with --headful."
        ),
    )
    parser.add_argument("--headful",          action="store_true", help="Show browser window")
    parser.add_argument("--list-types",       action="store_true", help="Print all stat types")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    print("Starting DraftKings Pick6 scraper (Playwright)...")
    try:
        main()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from e