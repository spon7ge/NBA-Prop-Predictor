"""
Run DFS / book scrapers in one shot; artifacts go under ``data/props/``:

  underdogs/     — Underdog pick'em JSON
  prizepicks/    — PrizePicks projections JSON
  pinnacle/      — Pinnacle matchups JSON (Selenium)
  player_lines/  — Odds-API.io player props JSON (FD/DK multi)
  team_lines/    — Odds-API.io spread + totals JSON

From repo root::

  python scripts/run_prop_scrapers.py
  python scripts/run_prop_scrapers.py --skip-pinnacle
  python scripts/run_prop_scrapers.py --only underdog,prizepicks,prop_odds,team_odds

Env: ``API_KEY_IO_1`` (or ``ODDS_API_IO_KEY``) for prop_odds / team_odds; ``.env`` loaded from repo root.

Note: ``src/utils/helpers.py`` still loads team odds from ``data/raw/team_lines`` and DFS/US CSVs from
``data/raw/player_lines``. This runner writes Odds-API.io *JSON* under ``data/props/``. Point loaders
or copy files if your pipeline expects the old paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROPS = PROJECT_ROOT / "data" / "props"


def _ensure_props_subdirs() -> None:
    for sub in (
        "underdogs",
        "prizepicks",
        "pinnacle",
        "player_lines",
        "team_lines",
    ):
        (DATA_PROPS / sub).mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def run_underdog() -> None:
    from src.scrapers.underdog import UnderdogScraper

    print("\n=== Underdog ===")
    scraper = UnderdogScraper()
    scraper.scrape()


def run_prizepicks(*, head: int = 0) -> None:
    from src.scrapers.prizepicks import PrizePicks_Scraper, print_rows_tsv

    print("\n=== PrizePicks ===")
    scraper = PrizePicks_Scraper()
    if head > 0:
        print_rows_tsv(scraper.lines, limit=head)


def run_pinnacle() -> None:
    from src.scrapers.pinnacle import PinnacleNBAScraper

    print("\n=== Pinnacle ===")
    PinnacleNBAScraper().run()


def run_prop_odds() -> None:
    from src.scrapers.prop_odds import OddsIoScraper, save_odds_io_pull_json

    print("\n=== prop_odds (Odds-API.io player props) ===")
    scraper = OddsIoScraper()
    payload = scraper.run()
    out_path = save_odds_io_pull_json(
        events=scraper.events,
        odds=payload,
        bookmakers=scraper.bookmakers,
        out_dir=DATA_PROPS / "player_lines",
    )
    print(f"saved {out_path}")


def run_team_odds() -> None:
    from src.scrapers.team_odds import (
        TeamOddsIoScraper,
        filter_odds_payloads_team_lines_only,
        save_team_lines_json,
    )

    print("\n=== team_odds (Odds-API.io Spread / Totals) ===")
    scraper = TeamOddsIoScraper()
    payload = scraper.run()
    team_only = filter_odds_payloads_team_lines_only(payload)
    out_path = save_team_lines_json(
        events=scraper.events,
        odds=payload,
        bookmakers=scraper.bookmakers,
        out_dir=DATA_PROPS / "team_lines",
    )
    print(
        f"events_fetched={len(scraper.events)} "
        f"team_line_payloads={len(team_only)}"
    )
    print(f"saved {out_path}")


_RUNNERS: dict[str, object] = {
    "underdog": run_underdog,
    "prizepicks": run_prizepicks,
    "pinnacle": run_pinnacle,
    "prop_odds": run_prop_odds,
    "team_odds": run_team_odds,
}


def _parse_only(raw: str | None) -> set[str] | None:
    if not raw or not raw.strip():
        return None
    names = {x.strip().lower() for x in raw.split(",") if x.strip()}
    bad = names - set(_RUNNERS)
    if bad:
        raise SystemExit(f"Unknown --only name(s): {bad}. Valid: {sorted(_RUNNERS)}")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run prop/DFS scrapers; outputs under data/props/.",
    )
    parser.add_argument(
        "--only",
        metavar="NAMES",
        help=f"Comma-separated subset: {','.join(sorted(_RUNNERS))}",
    )
    parser.add_argument("--skip-underdog", action="store_true")
    parser.add_argument("--skip-prizepicks", action="store_true")
    parser.add_argument("--skip-pinnacle", action="store_true")
    parser.add_argument("--skip-prop-odds", action="store_true")
    parser.add_argument("--skip-team-odds", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining scrapers after a failure",
    )
    parser.add_argument(
        "--prizepicks-head",
        type=int,
        default=0,
        metavar="N",
        help="Print first N PrizePicks rows as TSV (0 = skip)",
    )
    args = parser.parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)

    _load_dotenv()
    _ensure_props_subdirs()

    only = _parse_only(args.only)
    skip = set()
    if args.skip_underdog:
        skip.add("underdog")
    if args.skip_prizepicks:
        skip.add("prizepicks")
    if args.skip_pinnacle:
        skip.add("pinnacle")
    if args.skip_prop_odds:
        skip.add("prop_odds")
    if args.skip_team_odds:
        skip.add("team_odds")

    order = ["underdog", "prizepicks", "pinnacle", "prop_odds", "team_odds"]
    failed = []
    for key in order:
        if only is not None and key not in only:
            continue
        if key in skip:
            continue
        try:
            if key == "prizepicks":
                run_prizepicks(head=args.prizepicks_head)
            else:
                _RUNNERS[key]()
        except Exception as e:
            failed.append((key, e))
            print(f"✗ {key} failed: {e}", file=sys.stderr)
            if not args.continue_on_error:
                raise
    if failed:
        print("\nFailures:", ", ".join(k for k, _ in failed), file=sys.stderr)
        return 1
    print("\n✓ All requested scrapers finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
