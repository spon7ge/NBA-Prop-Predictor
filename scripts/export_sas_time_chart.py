"""Export SAS time leading/trailing chart for the NYK blog."""
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'images' / 'nyk' / 'sas-time-leading.png'
NYK_ID = 1610612752

con = duckdb.connect()
con.execute(f"CREATE TABLE pbp AS SELECT * FROM read_parquet('{ROOT / 'data/raw/pbp_stats/p26_analysis.parquet'}')")
con.execute(
    f"CREATE TABLE nyk_stats AS SELECT * FROM read_csv('{ROOT / 'data/raw/playoff_stats/P26.csv'}') "
    f"WHERE TEAM_ID = {NYK_ID}"
)

VIEW_DDL = """
CREATE OR REPLACE VIEW nyk_games AS SELECT DISTINCT gameId FROM pbp WHERE teamId = 1610612752;
CREATE OR REPLACE VIEW pbp_nyk AS SELECT p.* FROM pbp p INNER JOIN nyk_games g USING (gameId);
CREATE OR REPLACE VIEW game_meta AS SELECT DISTINCT LPAD(CAST(GAME_ID AS VARCHAR), 10, '0') AS gameId, GAME_ID, GAME_DATE, MATCHUP, WL, OPP_OPP_ABBREVIATION_base FROM nyk_stats;
CREATE OR REPLACE VIEW nyk_loc AS SELECT gameId, MAX(location) AS nyk_loc FROM pbp WHERE teamId = 1610612752 GROUP BY 1;
CREATE OR REPLACE VIEW games_enriched AS
WITH base AS (
    SELECT p.gameId, p.actionNumber, p.period,
        CAST(regexp_extract(p.clock, 'PT(\\d+)M', 1) AS INT) * 60 + CAST(regexp_extract(p.clock, 'M([\\d.]+)S', 1) AS DOUBLE) AS secs_remaining,
        l.nyk_loc,
        LAST_VALUE(TRY_CAST(p.scoreHome AS DOUBLE) IGNORE NULLS) OVER (PARTITION BY p.gameId ORDER BY p.actionNumber ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS score_home_n,
        LAST_VALUE(TRY_CAST(p.scoreAway AS DOUBLE) IGNORE NULLS) OVER (PARTITION BY p.gameId ORDER BY p.actionNumber ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS score_away_n
    FROM pbp_nyk p INNER JOIN nyk_loc l USING (gameId)
)
SELECT *,
    CASE WHEN nyk_loc = 'h' THEN COALESCE(score_home_n, 0) ELSE COALESCE(score_away_n, 0) END AS nyk_live,
    CASE WHEN nyk_loc = 'h' THEN COALESCE(score_away_n, 0) ELSE COALESCE(score_home_n, 0) END AS opp_live,
    CASE WHEN nyk_loc = 'h' THEN COALESCE(score_home_n, 0) ELSE COALESCE(score_away_n, 0) END
      - CASE WHEN nyk_loc = 'h' THEN COALESCE(score_away_n, 0) ELSE COALESCE(score_home_n, 0) END AS live_margin
FROM base;
"""
for stmt in VIEW_DDL.strip().split(';\n'):
    if stmt.strip():
        con.execute(stmt)

SAS_TIME_SQL = """
WITH sas_plays AS (
    SELECT g.gameId, g.actionNumber, g.period, g.secs_remaining, g.live_margin, m.GAME_DATE, m.MATCHUP, m.WL,
        CASE WHEN g.period <= 4 THEN (g.period - 1) * 720.0 + (720.0 - COALESCE(g.secs_remaining, 0))
            ELSE 2880.0 + (g.period - 5) * 300.0 + (300.0 - COALESCE(g.secs_remaining, 0)) END AS game_elapsed
    FROM games_enriched g
    JOIN game_meta m ON LPAD(CAST(g.gameId AS VARCHAR), 10, '0') = m.gameId
    WHERE m.OPP_OPP_ABBREVIATION_base = 'SAS'
),
windows AS (
    SELECT *, LEAD(game_elapsed) OVER (PARTITION BY gameId ORDER BY actionNumber) - game_elapsed AS dur_sec
    FROM sas_plays
)
SELECT gameId, MATCHUP, WL, MIN(GAME_DATE) AS game_date,
    ROUND(SUM(CASE WHEN live_margin > 0 AND dur_sec > 0 AND dur_sec <= 120 THEN dur_sec ELSE 0 END) / 60, 1) AS min_leading,
    ROUND(SUM(CASE WHEN live_margin < 0 AND dur_sec > 0 AND dur_sec <= 120 THEN dur_sec ELSE 0 END) / 60, 1) AS min_trailing,
    ROUND(SUM(CASE WHEN live_margin = 0 AND dur_sec > 0 AND dur_sec <= 120 THEN dur_sec ELSE 0 END) / 60, 1) AS min_tied
FROM windows GROUP BY 1, 2, 3 ORDER BY MIN(GAME_DATE)
"""

sas_time = con.execute(SAS_TIME_SQL).df()
sas_time['label'] = sas_time['MATCHUP'].str.replace('NYK ', '') + ' (' + sas_time['WL'] + ')'
COLORS = {'Leading': '#006BB6', 'Tied': '#9CA3AF', 'Trailing': '#CC0000'}

fig, (ax_games, ax_series) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [2.2, 1]})
sns.set_theme(style='whitegrid', context='notebook')

y = np.arange(len(sas_time))
h = 0.65
ax_games.barh(y, sas_time['min_leading'], height=h, label='Leading', color=COLORS['Leading'])
ax_games.barh(y, sas_time['min_tied'], height=h, left=sas_time['min_leading'], label='Tied', color=COLORS['Tied'])
ax_games.barh(
    y, sas_time['min_trailing'], height=h,
    left=sas_time['min_leading'] + sas_time['min_tied'],
    label='Trailing', color=COLORS['Trailing'],
)
for yi, row in enumerate(sas_time.itertuples()):
    total = row.min_leading + row.min_tied + row.min_trailing
    ax_games.text(total + 0.4, yi, f'{row.min_leading:.0f}↑ / {row.min_trailing:.0f}↓', va='center', fontsize=8)
ax_games.set_yticks(y)
ax_games.set_yticklabels(sas_time['label'])
ax_games.set_xlabel('Minutes of game clock')
ax_games.set_title('NYK Game Clock vs Spurs — Leading / Tied / Trailing')
ax_games.legend(loc='lower right', fontsize=9)
ax_games.set_xlim(0, sas_time[['min_leading', 'min_tied', 'min_trailing']].sum(axis=1).max() + 8)

series = sas_time[['min_leading', 'min_tied', 'min_trailing']].sum()
series_pct = (series / series.sum() * 100).round(1)
bars = ax_series.bar(
    ['Leading', 'Tied', 'Trailing'], series.values,
    color=[COLORS['Leading'], COLORS['Tied'], COLORS['Trailing']],
    edgecolor='white', width=0.55,
)
for bar, pct in zip(bars, series_pct):
    ax_series.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f'{pct:.0f}%', ha='center', fontweight='bold')
ax_series.set_ylabel('Total minutes (5 games)')
ax_series.set_title('Finals Series vs SAS')
ax_series.set_ylim(0, series.max() + 15)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Wrote {OUT.relative_to(ROOT)}')
