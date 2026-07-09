"""
HoopVista daily data pipeline.

Steps:
  1. Ingest APIs      — NBA stats, odds/props, Rotowire metadata
  2. Load raw tables  — merge raw.* into silver.player_gamelogs
  3. Transform        — bronze → silver → gold → ml (Python → Supabase)
  4. Generate predictions — write ml.predictions for the slate date

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

# ── Config (set via docker-compose / .env) ────────────────────────────────────

REPO_ROOT = os.environ.get("HOOPVISTA_REPO_ROOT", "")
NBA_SEASON = os.environ.get("HOOPVISTA_NBA_SEASON", "2025-26")
SEASON_TYPE = os.environ.get("HOOPVISTA_SEASON_TYPE", "Regular Season")
ROTOWIRE_SEASON = os.environ.get("HOOPVISTA_ROTOWIRE_SEASON", NBA_SEASON.split("-")[0])

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
    """Build a pipeline command executed from repo root."""
    inner = " ".join(parts)
    if REPO_ROOT:
        return f'cd "{REPO_ROOT}" && {RUNNER} {inner}'
    return f"{RUNNER} {inner}"


with DAG(
    dag_id="hoopvista_daily_pipeline",
    description="Ingest APIs → load tables → transforms → ML predictions",
    schedule="0 6,14,22 * * *",  # 3× daily (ET-ish; adjust TZ in Airflow config)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["hoopvista", "nba", "ml"],
    default_args=default_args,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # ── 1. Ingest APIs ───────────────────────────────────────────────────────
    with TaskGroup(group_id="ingest_apis", tooltip="Fetch external APIs") as ingest_apis:
        ingest_nba_stats = BashOperator(
            task_id="nba_stats",
            bash_command=_cmd(
                "python",
                "src/utils/playerLogs.py",
                "--league",
                "nba",
                f"--season {NBA_SEASON}",
                f'--season-type "{SEASON_TYPE}"',
                "--db-upsert",
                "--sequential",
            ),
        )

        ingest_odds_props = BashOperator(
            task_id="odds_props",
            bash_command=_cmd("python", "scripts/PropFinder.py"),
        )

        ingest_rotowire = BashOperator(
            task_id="rotowire",
            bash_command=_cmd(
                "python",
                "src/scrapers/rotowire_scraper.py",
                f"--season {ROTOWIRE_SEASON}",
            ),
        )

    # ── 2. Load raw tables (silver merge from raw.*) ───────────────────────────
    load_raw_tables = BashOperator(
        task_id="load_raw_tables",
        bash_command=_cmd(
            "python",
            "scripts/upload_silver.py",
            f"--season {NBA_SEASON}",
            f'--season-type "{SEASON_TYPE}"',
        ),
    )

    # ── 3. Materialize transforms (bronze/silver/gold/ml → Supabase) ───────────
    run_transforms = BashOperator(
        task_id="run_transforms",
        bash_command=_cmd("python", "scripts/run_transforms.py"),
    )

    transform_test = BashOperator(
        task_id="transform_test",
        bash_command=_cmd("python", "scripts/run_transforms.py", "--test", "ml"),
        trigger_rule="all_done",
    )

    # ── 4. Generate predictions ────────────────────────────────────────────────
    generate_predictions = BashOperator(
        task_id="generate_predictions",
        bash_command=_cmd(
            "python",
            "scripts/generate_predictions.py",
            "--prop all",
            "--game-date {{ ds }}",
            f'--season-type "{SEASON_TYPE}"',
        ),
    )

    # Runs gold_prediction_accuracy after yesterday's box scores have landed.
    score_predictions = BashOperator(
        task_id="score_predictions",
        bash_command=_cmd(
            "python",
            "scripts/run_transforms.py",
            "--select",
            "gold_prediction_accuracy",
        ),
    )

    start >> ingest_apis >> load_raw_tables >> run_transforms >> transform_test >> generate_predictions >> score_predictions >> end
