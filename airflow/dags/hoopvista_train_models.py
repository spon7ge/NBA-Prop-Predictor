"""
Weekly model retraining (separate from daily predictions).

Run after dbt ml features are fresh. Models are reused by generate_predictions.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO_ROOT = os.environ.get("HOOPVISTA_REPO_ROOT", "")
NBA_SEASON = os.environ.get("HOOPVISTA_NBA_SEASON", "2025-26")
SEASON_TYPE = os.environ.get("HOOPVISTA_SEASON_TYPE", "Regular Season")
RUNNER = "bash scripts/run_pipeline_step.sh"


def _cmd(*parts: str) -> str:
    inner = " ".join(parts)
    if REPO_ROOT:
        return f'cd "{REPO_ROOT}" && {RUNNER} {inner}'
    return f"{RUNNER} {inner}"


with DAG(
    dag_id="hoopvista_train_models",
    description="Retrain quantile models for min/ppm/rpm/apm",
    schedule="0 8 * * 1",  # Mondays 08:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["hoopvista", "ml", "training"],
    default_args={
        "owner": "hoopvista",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
) as dag:
    dbt_ml = BashOperator(
        task_id="dbt_ml_features",
        bash_command=_cmd("python", "scripts/run_dbt.py", "run", "--select", "ml"),
    )

    train_models = BashOperator(
        task_id="train_all_models",
        bash_command=_cmd(
            "python",
            "scripts/train_model.py",
            "--prop all",
            f"--season-year {NBA_SEASON}",
            f'--season-type "{SEASON_TYPE}"',
        ),
    )

    dbt_ml >> train_models
