# NBA Player Prop Predictor

This repo runs a **live prop model** (quantile-style projections and EV math) and ships results to a **static site** in `docs/`. The model scores book lines, builds **2-leg parlays**, and exports JSON that the browser loads—no API on the site itself.

### Top Pairs

- Ranks **2-leg parlays** by **expected value** using the **live model’s** projections, implied odds, and parlay math.
- **PrizePicks** and **Underdog** each have their own slate: toggle the book to swap between `prizepicks.json` and `underdog.json` under `data/props/ev_analysis/` (the deploy workflow copies them into `docs/data/...`; see `.github/workflows/pages.yml`).
- Each card shows EV, hit probability, Kelly, both legs (line, side, model vs line, form, matchup, defense rank, L10 hit rate), plus game total and spread context.

### All Players

- Lists **every modeled line**, not just pairs: **`all_line_probs.json`** (JSON Lines—one object per row).
- **PrizePicks** and **Underdog** can both appear for the same player/market (`LINE_BOOKMAKER`); the **Book** column uses logo assets (`prizepicks.webp`, `underdog.webp`).
- Search filters by player name; the table includes O/U odds, EV over/under, **Best** side, and L5/L10/L15 hit rates aligned to that best side.

---

## Features

- **Live model** driving projections and EV for props and 2-leg parlays
- **Public site:** **Top Pairs** (per bookmaker) + **All Players** tab with search
- **Dual bookmaker** support on the site (PrizePicks & Underdog JSON slates)
- Historical prop tables, rolling stats, usage, matchup, and team context (notebooks / exports)
- Kelly Criterion and parlay tooling in the analysis pipeline

## Tech Stack

- **XGBoost** (minutes and per-minute work in progress)
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
├── data/                # Training data + ev_analysis JSON consumed by the site
└── docs/                # Static website (GitHub Pages artifact)
```

## Disclaimer

This project is for educational purposes only. Sports betting involves risk. Always gamble responsibly.
