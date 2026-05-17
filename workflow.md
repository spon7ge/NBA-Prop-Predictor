# HoopVista — End-to-End Workflow

## Overview

HoopVista is an NBA player prop prediction system. On game day, three scripts run in sequence to produce DFS slates for PrizePicks, Underdog, DraftKings Pick6, and Betr DFS.

---

## Game-Day Order

```
scripts/fetch_gamelogs.py  →  scripts/PropFinder.py  →  live.py
```

---

## Step 1 — Update Player Stats (`scripts/fetch_gamelogs.py`)

Pulls the latest NBA game logs from the NBA API and merges supporting data to keep `base_df` current before predictions run.

**What it does:**
1. Fetches box scores via `NBAGameLogs` (skips game IDs already in the checkpoint at `data/raw/cache/tracking_checkpoint.csv`).
2. Merges player positions from `data/raw/player_positions/nba_2026_players.csv`.
3. Applies bookmaker name canonicalization against the reference roster in `data/raw/season_stats/S26.csv`.
4. Scrapes Rotowire (Playwright, headless) for team spreads and totals → merges `TEAM_SPREAD` / `GAME_TOTAL` onto every row.
5. Writes output to `data/raw/playoff_stats/P26.csv`. → If regular season it saves in `data/raw/season_stats/`

**Flags:**
```bash
python scripts/fetch_gamelogs.py                          # full run
python scripts/fetch_gamelogs.py --skip-rotowire          # reuse existing Rotowire CSV
python scripts/fetch_gamelogs.py --skip-nba-fetch \
    --parquet data/raw/playoff_stats/_last_fetch.parquet  # skip NBA API, load cached frame
```

---

## Step 2 — Fetch Prop Lines (`scripts/PropFinder.py`)

Pulls today's prop lines from the Odds API and writes two CSV files to `data/raw/player_lines/`:

| File pattern | Contents |
|---|---|
| `NBA_DFS_<YYYYMMDD>*.csv` | DFS platform lines (PrizePicks, Underdog, DK Pick6, Betr DFS) |
| `NBA_US_<YYYYMMDD>*.csv` | US sharp/bookmaker lines with odds (Pinnacle, FanDuel, DraftKings, BetMGM, etc.) |

```bash
python scripts/PropFinder.py
```

> `scripts/run_scrapers.py` contains the original custom scrapers but is **not in use**. PropFinder.py via the Odds API is the active source.

---

## Step 3 — Run the Daily Pipeline (`live.py`)

Orchestrates everything from lineup scraping through slate output.

```bash
python live.py
```

### 3a. Scrape Lineups & Injuries
`NBADailyLineups` scrapes Rotowire's NBA Lineups page for today's starters and out players. `outPlayers` is populated globally so downstream context adjustments know who is missing.

### 3b. Load Data
- `load_base_df()` — concatenates S25 + P25 + S26 + P26 CSVs; runs `_detect_star_players` to flag team stars on each combined frame.
- `load_team_odds()` — latest `NBA_*.json` from `data/raw/team_lines/` (spreads + totals).
- `load_player_lines(today_str)` — reads today's `NBA_DFS_*.csv` and `NBA_US_*.csv`; splits each into points / assists / rebounds sub-frames.

### 3c. Load Models
`load_models()` loads the most recent `.joblib` bundle for each sub-model from `src/models/saved_models/`:

| Bundle prefix | Predicts |
|---|---|
| `min_quantile_xgb` | Minutes played (MIN) |
| `ppm_quantile_xgb` | Points per minute (PPM) |
| `apm_quantile_xgb` | Assists per minute (APM) |
| `rpm_quantile_xgb` | Rebounds per minute (RPM) |

Each bundle contains `{ "q_0.10": model, "q_0.50": model, "q_0.90": model, "feature_names": [...] }`.

### 3d. Predict Quantiles
`predict_min_times_rate()` is called once per stat category (PTS, AST, REB):
1. Runs `min_pipeline` → Q10/Q50/Q90 minutes per player.
2. Runs the matching rate pipeline (PPM / APM / RPM) → Q10/Q50/Q90 rate per player.
3. Multiplies `MIN × RATE` to get `STAT` quantiles: `STAT_Q10 / STAT_Q50 / STAT_Q90`.

Only players who appear in the DFS lines for that category are processed.

### 3e. Context Adjustments
`get_game_context()` + `adjust_predictions()` shift the raw quantiles based on game-day signals:

| Signal | MIN weight | RATE weight |
|---|---|---|
| Star player absent | 1.00 | 1.00 |
| Pace matchup | 0.50 | 0.35 |
| Spread / blowout risk | 0.45 | 0.15 |
| Defensive matchup | 0.35 | 0.40 |

- Uses Bayesian shrinkage (`tau = 20`) so small samples barely move projections.
- Rate shifts are capped at ±12%.
- Quantiles are coerced to remain non-negative and monotone (Q10 ≤ Q50 ≤ Q90) after adjustment.

### 3f. Simulate Line Probabilities
`line_probs_for_market()` runs Monte Carlo simulation (default 10,000 sims) for each player / DFS line:
- **PTS** → `run_pts_simulation` (triangular minutes × triangular PPM, continuous).
- **AST / REB** → `run_count_simulation_nbinom` (Negative-Binomial, overdispersion-aware).

Produces `P_OVER` and `P_UNDER` for every player-line combination across all DFS platforms.

### 3g. Enrich DFS Picks
`enrich_dfs_picks()` (from `src/utils/generalized_best_bets_v2.py`) cross-references model probabilities against sharp US bookmaker lines:

Assigns each pick a **tier**:

| Tier | Meaning |
|---|---|
| `sharp_verified` | A sharp book (Pinnacle / FD / DK) agrees with the model's lean |
| `dfs_only` | No sharp coverage; model agrees with the DFS platform's no-vig line |
| `conflict` | Model lean disagrees with sharp lean |
| `no_model` | No model row for this pick (line gap too large or unsupported market) |

Also attaches game context (spread, total) from the Circa/BetOnline team-lines JSON if available.

Outputs:
- `data/props/enriched/` — enriched picks CSV
- `data/props/ev_analysis/dfs_sharp_aligned_<date>.json` — aligned JSON used by slate builder

### 3h. Build Slates
`build_dfs_slates_from_aligned()` is called once per bookmaker and generates 2-, 3-, 5-, and 6-leg DFS entries:

**Platform payouts:**

| Platform | 2-leg | 3-leg | 5-leg | 6-leg |
|---|---|---|---|---|
| PrizePicks / Underdog / Betr DFS | 2.0× | 4.5× | 19.0× | 36.5× |
| DraftKings Pick6 | 2.0× | 4.0× | 19.0× | 24.0× |

**Selection logic:**
1. Filters to high-total games and valid N-leg combinations.
2. Tiers candidates by combined model probability and EV.
3. Greedy selection across tiers; 5/6-leg enumeration capped at 24 unique players.

Output JSON files written to `data/props/ev_analysis/`:
```
prizepicks.json       prizepicks_3leg.json  prizepicks_5leg.json  prizepicks_6leg.json
underdog.json         underdog_3leg.json    underdog_5leg.json    underdog_6leg.json
betr.json             betr_3leg.json        betr_5leg.json        betr_6leg.json
draftKings.json       draftKings_3leg.json  draftKings_5leg.json  draftKings_6leg.json
```

### 3i. Log to Ledger
`log.snapshot()` appends today's predictions and slate paths to the ledger for tracking and future result grading. Model columns are flattened (`P_OVER`, `P_UNDER`, `MIN_Q10/Q50/Q90`, `STAT_Q10/Q50/Q90`) before writing.

---

## Model Training (Periodic, Not Daily)

Training is done manually in the notebooks in `src/models/` — typically at the start of a new season phase (e.g., beginning of playoffs).

```
src/models/min_quantile_model.ipynb   ← minutes model
src/models/ppm_quantile_model.ipynb   ← points per minute
src/models/apm_quantile_model.ipynb   ← assists per minute
src/models/rpm_quantile_model.ipynb   ← rebounds per minute
```

Key training decisions:
- **Time-series split / walk-forward validation** — prevents temporal leakage; no random k-fold shuffling.
- **Star-absence flags** — `main_star_active`, `second_star_active`, `third_star_active` tell the model who was missing.
- **Separate models for stars vs. role players** — avoids regression-to-the-mean bias across usage tiers.
- **Rolling averages are pre-computed up to the game date only** — no look-ahead.

Saved bundles land in `src/models/saved_models/`; `live.py` always loads the latest by filename sort.

---

## Data Flow Summary

```
NBA API + Rotowire
       │
       ▼
scripts/fetch_gamelogs.py
       │  data/raw/playoff_stats/P26.csv
       │  data/raw/rotowire/rotowire_nba_2025.csv
       ▼
Odds API (DFS + US regions)
       │
       ▼
scripts/PropFinder.py
       │  data/raw/player_lines/NBA_DFS_<date>.csv
       │  data/raw/player_lines/NBA_US_<date>.csv
       ▼
live.py
  ├── 3a  Rotowire lineup scrape → outPlayers dict
  ├── 3b  load_base_df() + load_team_odds() + load_player_lines()
  ├── 3c  load_models() → min / ppm / apm / rpm XGB bundles
  ├── 3d  predict_min_times_rate() → STAT_Q10/Q50/Q90 (PTS, AST, REB)
  ├── 3e  context adjustments (stars, pace, spread, def)
  ├── 3f  Monte Carlo simulation → P_OVER / P_UNDER per DFS line
  ├── 3g  enrich_dfs_picks() → sharp-verified tiers + dfs_sharp_aligned JSON
  ├── 3h  build_dfs_slates_from_aligned() → 2/3/5/6-leg JSON per bookmaker
  └── 3i  log.snapshot() → ledger
```
