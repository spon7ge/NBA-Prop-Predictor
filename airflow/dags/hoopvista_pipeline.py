"""
HoopVista daily data pipeline.

Steps:
  1. Ingest APIs — NBA/WNBA stats → raw.*, odds/props (PropFinder)
  2. Build silver  — merge raw → silver.nba_* / silver.wnba_*
  3. Live ML       — live prop predictions + greedy multi-leg slates → ml.*

Gold feature tables and model retraining are manual (not scheduled).

Requires env (repo-root .env or Airflow Variables):
  SUPABASE_DB_URL, API_KEY, API_KEY_IO_1/2 (as needed)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

REPO_ROOT = os.environ.get("HOOPVISTA_REPO_ROOT", "")
NBA_SEASON = os.environ.get("HOOPVISTA_NBA_SEASON", "2025-26")
WNBA_SEASON = os.environ.get("HOOPVISTA_WNBA_SEASON", "2026")
SEASON_TYPE = os.environ.get("HOOPVISTA_SEASON_TYPE", "Regular Season")
PROPFINDER_LEAGUE = os.environ.get("HOOPVISTA_PROPFINDER_LEAGUE", "wnba")
# Comma-separated: nba, wnba, or both (default both)
_LIVE_LEAGUES_RAW = os.environ.get("HOOPVISTA_LIVE_LEAGUES", "nba,wnba")
LIVE_LEAGUES = [
    x.strip().lower()
    for x in _LIVE_LEAGUES_RAW.split(",")
    if x.strip().lower() in ("nba", "wnba")
] or ["wnba"]

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


with DAG(
    dag_id="hoopvista_daily_pipeline",
    description=(
        "Fetch raw (NBA+WNBA) → PropFinder → silver → live props + slates"
    ),
    schedule="0 6,14,22 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["hoopvista", "nba", "wnba", "ingest", "silver", "live"],
    default_args=default_args,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    with TaskGroup(group_id="ingest_apis", tooltip="Fetch external APIs") as ingest_apis:
        BashOperator(
            task_id="nba_stats",
            bash_command=_cmd(
                "python",
                "scripts/fetch_raw.py",
                "--league",
                "nba",
                f"--season {NBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
                "--raw-only",
                "--sequential",
            ),
        )

        BashOperator(
            task_id="wnba_stats",
            bash_command=_cmd(
                "python",
                "scripts/fetch_raw.py",
                "--league",
                "wnba",
                f"--season {WNBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
                "--raw-only",
                "--sequential",
            ),
        )

        BashOperator(
            task_id="odds_props",
            bash_command=_cmd(
                "python",
                "scripts/PropFinder.py",
                f"--league {PROPFINDER_LEAGUE}",
            ),
        )

    with TaskGroup(group_id="build_silver", tooltip="Merge raw → silver") as build_silver:
        BashOperator(
            task_id="nba",
            bash_command=_cmd(
                "python",
                "scripts/fetch_raw.py",
                "--league",
                "nba",
                f"--season {NBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
                "--silver-only",
            ),
        )

        BashOperator(
            task_id="wnba",
            bash_command=_cmd(
                "python",
                "scripts/fetch_raw.py",
                "--league",
                "wnba",
                f"--season {WNBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
                "--silver-only",
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

    start >> ingest_apis >> build_silver >> live_ml >> end
