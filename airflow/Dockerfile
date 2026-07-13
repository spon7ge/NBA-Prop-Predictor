FROM apache/airflow:2.10.4-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    && playwright install --with-deps chromium

ENV HOOPVISTA_REPO_ROOT=/opt/airflow/hoopvista \
    HOOPVISTA_NBA_SEASON=2025-26 \
    HOOPVISTA_WNBA_SEASON=2025 \
    HOOPVISTA_SEASON_TYPE="Regular Season" \
    HOOPVISTA_PROPFINDER_LEAGUE=wnba \
    PYTHONPATH=/opt/airflow/hoopvista
