"""
HoopVista Airflow DAGs.

1. ``hoopvista_daily_stats`` — midnight local
   NBA/WNBA ``fetch`` → ``clean`` → ``grade_live_props`` (raw + silver + grades)

2. ``hoopvista_live_odds`` — 8:00, 12:00, 15:00 local
   PropFinder ∥ RotoWire starters → live props → live slates

Timezone: ``HOOPVISTA_TZ`` (default ``America/Los_Angeles``).

Requires env (repo-root .env or Airflow Variables):
  SUPABASE_DB_URL, API_KEY, API_KEY_IO_1/2 (as needed)
"""

from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

REPO_ROOT = os.environ.get("HOOPVISTA_REPO_ROOT", "")
NBA_SEASON = os.environ.get("HOOPVISTA_NBA_SEASON", "2025-26")
WNBA_SEASON = os.environ.get("HOOPVISTA_WNBA_SEASON", "2026")
SEASON_TYPE = os.environ.get("HOOPVISTA_SEASON_TYPE", "Regular Season")
PROPFINDER_LEAGUE = os.environ.get("HOOPVISTA_PROPFINDER_LEAGUE", "wnba")
_LIVE_LEAGUES_RAW = os.environ.get("HOOPVISTA_LIVE_LEAGUES", "nba,wnba")
LIVE_LEAGUES = [
    x.strip().lower()
    for x in _LIVE_LEAGUES_RAW.split(",")
    if x.strip().lower() in ("nba", "wnba")
] or ["wnba"]
# RotoWire starters: scrape every live league (or both via ``all``).
STARTERS_LEAGUE = (
    "all"
    if set(LIVE_LEAGUES) >= {"nba", "wnba"}
    else LIVE_LEAGUES[0]
)

TZ_NAME = os.environ.get("HOOPVISTA_TZ", "America/Los_Angeles")
TZ = pendulum.timezone(TZ_NAME)
START = pendulum.datetime(2026, 1, 1, tz=TZ)

RUNNER = "bash scripts/run_pipeline_step.sh"

default_args = {
    "owner": "hoopvista",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _cmd(*parts: str) -> str:
    inner = " ".join(parts)
    if REPO_ROOT:
        return f'cd "{REPO_ROOT}" && {RUNNER} {inner}'
    return f"{RUNNER} {inner}"


# ── Midnight: game logs → silver ─────────────────────────────────────────────

with DAG(
    dag_id="hoopvista_daily_stats",
    description="Midnight fetch + silver + grade live props (NBA/WNBA)",
    schedule="0 0 * * *",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    tags=["hoopvista", "nba", "wnba", "fetch", "silver", "grades"],
    default_args=default_args,
) as daily_stats:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    with TaskGroup(group_id="fetch_raw", tooltip="NBA/WNBA stats → raw.*") as fetch_raw:
        BashOperator(
            task_id="nba",
            bash_command=_cmd(
                "python",
                "-m",
                "src.pipeline.fetch",
                "--league",
                "nba",
                f"--season {NBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
                "--sequential",
            ),
        )
        BashOperator(
            task_id="wnba",
            bash_command=_cmd(
                "python",
                "-m",
                "src.pipeline.fetch",
                "--league",
                "wnba",
                f"--season {WNBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
                "--sequential",
            ),
        )

    with TaskGroup(group_id="build_silver", tooltip="raw.* → silver.*") as build_silver:
        BashOperator(
            task_id="nba",
            bash_command=_cmd(
                "python",
                "-m",
                "src.pipeline.clean",
                "--league",
                "nba",
                f"--season {NBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
            ),
        )
        BashOperator(
            task_id="wnba",
            bash_command=_cmd(
                "python",
                "-m",
                "src.pipeline.clean",
                "--league",
                "wnba",
                f"--season {WNBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
            ),
        )

    with TaskGroup(
        group_id="grade_live_props",
        tooltip="Score last night's live props vs silver box scores",
    ) as grade_live_props:
        for league in LIVE_LEAGUES:
            BashOperator(
                task_id=league,
                bash_command=_cmd(
                    "python",
                    "scripts/grade_live_props.py",
                    "--league",
                    league,
                    "--lookback-days",
                    "3",
                ),
            )

    start >> fetch_raw >> build_silver >> grade_live_props >> end


# ── Daytime: odds → live predictions ─────────────────────────────────────────

with DAG(
    dag_id="hoopvista_live_odds",
    description="Odds + starters + live props/slates (8am / 12pm / 3pm local)",
    schedule="0 8,12,15 * * *",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    tags=["hoopvista", "odds", "live", "props", "slates", "starters"],
    default_args=default_args,
) as live_odds:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    odds_props = BashOperator(
        task_id="odds_props",
        bash_command=_cmd(
            "python",
            "scripts/PropFinder.py",
            f"--league {PROPFINDER_LEAGUE}",
        ),
    )

    scrape_starters = BashOperator(
        task_id="scrape_starters",
        bash_command=_cmd(
            "python",
            "-m",
            "src.scrapers.rotowire_starters_scraper",
            f"--league {STARTERS_LEAGUE}",
            "--update",
        ),
    )

    with TaskGroup(
        group_id="live_ml",
        tooltip="Live prop predictions + greedy multi-leg slates",
    ) as live_ml:
        with TaskGroup(
            group_id="live_props",
            tooltip="Upsert ml.*_live_prop_predictions",
        ) as live_props:
            for league in LIVE_LEAGUES:
                BashOperator(
                    task_id=league,
                    bash_command=_cmd(
                        "python",
                        "scripts/run_live_props.py",
                        "--league",
                        league,
                        f'--season-type "{SEASON_TYPE}"',
                    ),
                )

        with TaskGroup(
            group_id="live_slates",
            tooltip="Upsert ml.*_live_slates (Top Legs parlays)",
        ) as live_slates:
            for league in LIVE_LEAGUES:
                BashOperator(
                    task_id=league,
                    bash_command=_cmd(
                        "python",
                        "scripts/run_live_slates.py",
                        "--league",
                        league,
                        f'--season-type "{SEASON_TYPE}"',
                    ),
                )

        live_props >> live_slates

    start >> [odds_props, scrape_starters] >> live_ml >> end
