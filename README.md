# NBA Player Prop Predictor

Tools for NBA player prop analysis and expected value. The live site focuses on **historical context** for each prop: rolling performance, usage, matchup and team factors, and how those line up with book prices.

Modeling is evolving toward **per-minute rates** (points, assists, rebounds) combined with a **minutes** forecast.

## What the website shows (today)

The dashboard surfaces tables in the same spirit as the bookmaker analysis pipeline—player and prop context with recent form and market edges. Typical columns include:

- **Prop & market:** player, position, prop type, line, opponent, over/under odds, implied probabilities, EV over/under
- **Recent form:** average/median/std of the stat over the last 10 games, z-score, model prob over/under, hit rates over last 5 / 10 / 15 games
- **Role & usage:** average minutes and usage (and variability) over recent games
- **Matchup:** average stat vs this opponent, games in sample
- **Game & opponent context:** spread, total, opponent defensive rating/rank, opponent pace/rank

(See `src/historical_analysis/bookmakers.ipynb` for the full column set and export flow.)

## Roadmap: minutes + per-minute models

- **In progress:** an **XGBoost** model to predict **minutes**, paired with a **per-minute** model (e.g. points per minute) so projected stat totals combine role (minutes) and efficiency (rate).
- **Goal:** extend the same pattern to **points, assists, and rebounds** on a per-minute basis, then combine with predicted minutes for full stat projections.

Earlier experiments used **NGBoost** and normal-distribution EV math in notebooks and pipelines; the direction above is the current modeling focus.

## Features

- Historical prop tables with rolling stats, usage, matchup, and team context
- Expected value vs book/DFS lines (where wired into the pipeline)
- Kelly Criterion bet sizing (in analysis flows)
- Support for 2-leg and 3-leg parlays (in tooling)
- Interactive dashboard

## Tech Stack

- **ML:** NGBoost (legacy/experiments), **XGBoost** (minutes and per-minute work in progress)
- Normal distribution for probability estimation where used
- HTML/CSS/JS for visualization
- NBA API and The Odds API for data

## Project Structure

```
├── src/
│   ├── models/          # Model training and inference
│   ├── features/        # Feature engineering
│   ├── pipeline/        # Pipelines used for live predicting
│   └── scrapers/        # Data scraping modules
├── notebooks/           # Analysis and exploration
├── data/                # Training and prop data
└── docs/                # Dashboard files
```

## How it works (high level)

1. Ingest player stats, team metrics, and betting lines.
2. Engineer features from history and game context.
3. **Today:** rank and display historical + market context (EV where computed).
4. **Next:** predict minutes (XGBoost) and per-minute rates, then combine for PTS/AST/REB projections.

## Usage

The dashboard highlights top EV parlay opportunities and hit-rate style historical views, depending on which export/pipeline you run.

## Disclaimer

This project is for educational purposes only. Sports betting involves risk. Always gamble responsibly.
