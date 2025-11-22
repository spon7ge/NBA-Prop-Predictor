# NBA Player Prop Predictor

I built this NGBoost Model to help my friends and I stop making decisions based on vibes and hot takes, so I created a tool that actually quantifies edge using probabilistic modeling. Now instead of going off vibes, we can see which props have a positive EV and make data-driven decisions.

## Features

- **Player Performance Prediction**: NGBoost model trained on historical data, game context, volatility, and opponent metrics to create mean and variance models
- **Expected Value Calculation**: Using a normal distribution to find the probability of the prop hitting
- **Smart Bet Sizing**: Kelly Criterion for optimal bankroll management
- **Multiple Bet Types**: Single bets, 2-leg parlays, and 3-leg parlays
- **Dashboard**: Interactive web app for daily betting opportunities

## Tech Stack

- **Machine Learning**: NGBoost
- **Distribution**: Normal
- **Visualization**: HTML/CSS/JS
- **Data Sources**: NBA API, The Odds API

## Project Structure

```
├── MODELS/                   # Model training and inference
├── NOTEBOOKS/                # EV calculation and analysis
├── PRODUCTION/                # Where I calculate my EVs before displaying
├── BACKTEST/                 # Backtesting framework
├── FEATURES/                 # Feature engineering
├── NBAPropFinder/           # Data scraping modules
└── DATA/                     # Training and prop data
```

## How It Works

1. **Data Collection**: Scrapes player stats, team metrics, and betting lines from multiple sources
2. **Feature Engineering**: Builds comprehensive feature sets including recent form, matchups, and game context
3. **Model Training**: Trains NGBoost models to predict mean and variance of player performance distributions
4. **EV Calculation**: Uses analytical normal distribution calculations to estimate probabilities and calculate expected value
5. **Bet Selection**: Ranks opportunities by EV and provides Kelly-optimal bet sizing

## Usage

The Streamlit dashboard displays:

- Top EV single bets
- 2-leg and 3-leg parlay opportunities
- Over/under hit rates
- Confidence intervals and uncertainty metrics

## Disclaimer

This project is for educational purposes only. Sports betting involves risk, and there are too many variables to fully predict a player's outcome. Always gamble responsibly.
