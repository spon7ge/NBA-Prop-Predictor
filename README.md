# 🏀 NBA Player Prop Prediction & Betting Strategy
## 📌 Project Overview
This project is an end-to-end pipeline that predicts NBA player performance using players historical data, advanced stats to create ML models for each prop. Then grabbing the expected value (EV) and bet sizing using the Kelly Criterion for various betting scenarios (single legs, 2-leg, 3-leg props).
## Goals
### Predict NBA Player Prop Outcomes
Train machine learning models (e.g., XGBoost) to forecast player stat lines like Points, Assists, and Rebounds based on historical and contextual data.

### Build a Scalable Data Pipeline
- Create an automated pipeline to collect, clean, and store
- Historical player stats
- Opponent and team metrics
- Betting lines (PrizePicks, Underdog, sportsbooks)

### Calculate Smart Bets Using Probability & EV
- Evaluate bets using:
- Expected Value (EV)
- Kelly Criterion
- Blended probabilities (Z-Score + Monte Carlo) soon implement using machine learning

### Deploy a Streamlit Dashboard
- Display the best bets of the day, including:
- Predicted player stats from the model
- Over/Under recommendation
- EV rankings
- Parlay builder (2-leg, 3-leg, 4-leg props)
