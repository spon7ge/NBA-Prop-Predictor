# NBA Player Prop Predictor

NGBoost model for predicting NBA player props and calculating expected value. Compares PrizePicks and Underdog lines to sharp bookmaker odds to identify positive EV opportunities.

## Features

- Player performance prediction using NGBoost
- Expected value calculation with normal distribution
- Kelly Criterion bet sizing
- Support for 2-leg and 3-leg parlays
- Interactive dashboard

## Tech Stack

- NGBoost for machine learning
- Normal distribution for probability estimation
- HTML/CSS/JS for visualization
- NBA API and The Odds API for data

## Project Structure

```
├── src/
│   ├── models/          # Model training and inference
│   ├── features/        # Feature engineering
│   ├── pipeline/        # EV calculation pipeline
│   ├── backtest/        # Backtesting framework
│   └── scrapers/        # Data scraping modules
├── notebooks/           # Analysis and exploration
├── data/                # Training and prop data
└── docs/                # Dashboard files
```

## How It Works

1. Scrape player stats, team metrics, and betting lines
2. Engineer features from historical data and game context
3. Train NGBoost models to predict mean and variance
4. Calculate expected value using normal distribution
5. Rank opportunities by EV with Kelly-optimal bet sizing

## Usage

The dashboard displays top EV parlay opportunitie and hit rates.

## Disclaimer

This project is for educational purposes only. Sports betting involves risk. Always gamble responsibly.
