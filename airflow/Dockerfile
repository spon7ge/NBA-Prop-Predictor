FROM apache/airflow:2.10.4-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

# Install Python deps as the airflow user; browser system deps need root.
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt

USER root
RUN playwright install-deps chromium \
    && playwright install chromium \
    && mkdir -p /home/airflow/.cache/ms-playwright \
    && cp -a /root/.cache/ms-playwright/. /home/airflow/.cache/ms-playwright/ \
    && chown -R airflow:root /home/airflow/.cache/ms-playwright

USER airflow

ENV HOOPVISTA_REPO_ROOT=/opt/airflow/hoopvista \
    HOOPVISTA_NBA_SEASON=2025-26 \
    HOOPVISTA_WNBA_SEASON=2026 \
    HOOPVISTA_SEASON_TYPE="Regular Season" \
    HOOPVISTA_PROPFINDER_LEAGUE=wnba \
    HOOPVISTA_LIVE_LEAGUES=nba,wnba \
    PYTHONPATH=/opt/airflow/hoopvista \
    PLAYWRIGHT_BROWSERS_PATH=/home/airflow/.cache/ms-playwright
