"""
Run DFS / book scrapers in one shot; artifacts go under ``data/props/``:

  underdogs/        — Underdog pick'em JSON
  prizepicks/       — PrizePicks projections JSON
  pinnacle/         — Pinnacle matchups JSON (Selenium)
  dk+fd_props/       — Odds-API.io FanDuel + DraftKings player props (``draftkings_fanduel.py``)
  dk+fd_team_lines/  — Odds-API.io FanDuel + DraftKings spreads + totals (``draftkings_fanduel_team.py``)
  365+mgm_props/     — Odds-API.io BetMGM + Bet365 player props (``betmgm_bet365.py``)
  365+mgm_team_lines/ — Odds-API.io BetMGM + Bet365 spreads + totals (``betmgm_bet365_team.py``)

From repo root::

  python scripts/run_scrapers.py
  python scripts/run_scrapers.py --skip-pinnacle
  python scripts/run_scrapers.py --only underdog,prizepicks,dk_fd_props,dk_fd_team,mgm_bet365_props,mgm_bet365_team

Env:
  ``API_KEY_IO_1`` (or ``ODDS_API_IO_KEY``) for ``draftkings_fanduel`` and ``draftkings_fanduel_team``;
  ``API_KEY_IO_2`` (or ``ODDS_API_IO_KEY``) for ``betmgm_bet365`` and ``betmgm_bet365_team``.
  ``.env`` is loaded from repo root.

Note: ``src/utils/helpers.py`` may still point at ``data/raw/`` paths. Copy artifacts or update
loaders if the pipeline expects the old locations.
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
        "dk+fd_props",
        "dk+fd_team_lines",
        "365+mgm_props",
        "365+mgm_team_lines",
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


def run_dk_fd_props() -> None:
    from src.scrapers.draftkings_fanduel import OddsIoScraper, save_odds_io_pull_json

    print("\n=== draftkings_fanduel (Odds-API.io FD/DK player props) ===")
    scraper = OddsIoScraper()
    payload = scraper.run()
    out_path = save_odds_io_pull_json(
        events=scraper.events,
        odds=payload,
        bookmakers=scraper.bookmakers,
        out_dir=DATA_PROPS / "dk+fd_props",
    )
    print(f"saved {out_path}")


def run_dk_fd_team() -> None:
    from src.scrapers.draftkings_fanduel_team import fetchTeamLines

    print("\n=== draftkings_fanduel_team (Odds-API.io FD/DK spreads/totals) ===")
    _, path = fetchTeamLines(save_dir=DATA_PROPS / "dk+fd_team_lines")
    if not path:
        raise ValueError(
            "draftkings_fanduel_team failed (set API_KEY_IO_1 or ODDS_API_IO_KEY, check logs)."
        )
    print(f"saved {path}")


def run_mgm_bet365_props() -> None:
    from src.scrapers.betmgm_bet365 import OddsIoScraper, save_odds_io_pull_json

    print("\n=== betmgm_bet365 (Odds-API.io BetMGM/Bet365 player props) ===")
    scraper = OddsIoScraper()
    payload = scraper.run()
    out_path = save_odds_io_pull_json(
        events=scraper.events,
        odds=payload,
        bookmakers=scraper.bookmakers,
        out_dir=DATA_PROPS / "365+mgm_props",
    )
    print(f"saved {out_path}")


def run_mgm_bet365_team() -> None:
    from src.scrapers.betmgm_bet365_team import fetchTeamLines

    print("\n=== betmgm_bet365_team (Odds-API.io BetMGM/Bet365 spreads/totals) ===")
    key = os.environ.get("API_KEY_IO_2") or os.environ.get("ODDS_API_IO_KEY")
    if not key:
        raise ValueError(
            "Set API_KEY_IO_2 (or ODDS_API_IO_KEY) for betmgm_bet365_team."
        )
    _, path = fetchTeamLines(key, save_dir=DATA_PROPS / "365+mgm_team_lines")
    if path:
        print(f"saved {path}")


_RUNNERS: dict[str, object] = {
    "underdog": run_underdog,
    "prizepicks": run_prizepicks,
    "pinnacle": run_pinnacle,
    "dk_fd_props": run_dk_fd_props,
    "dk_fd_team": run_dk_fd_team,
    "mgm_bet365_props": run_mgm_bet365_props,
    "mgm_bet365_team": run_mgm_bet365_team,
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
    parser.add_argument("--skip-dk-fd-props", action="store_true")
    parser.add_argument("--skip-dk-fd-team", action="store_true")
    parser.add_argument("--skip-mgm-bet365-props", action="store_true")
    parser.add_argument("--skip-mgm-bet365-team", action="store_true")
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
    skip: set[str] = set()
    if args.skip_underdog:
        skip.add("underdog")
    if args.skip_prizepicks:
        skip.add("prizepicks")
    if args.skip_pinnacle:
        skip.add("pinnacle")
    if args.skip_dk_fd_props:
        skip.add("dk_fd_props")
    if args.skip_dk_fd_team:
        skip.add("dk_fd_team")
    if args.skip_mgm_bet365_props:
        skip.add("mgm_bet365_props")
    if args.skip_mgm_bet365_team:
        skip.add("mgm_bet365_team")

    order = [
        "underdog",
        "prizepicks",
        "pinnacle",
        "dk_fd_props",
        "dk_fd_team",
        "mgm_bet365_props",
        "mgm_bet365_team",
    ]
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
