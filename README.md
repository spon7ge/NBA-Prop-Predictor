# 🏀 NBA Player Prop Prediction & Betting Strategy
## Project Overview
This project is an end-to-end pipeline that predicts NBA player performance using players historical data, advanced stats to create ML models for each prop. Then grabbing the expected value (EV) and bet sizing using the Kelly Criterion for various betting scenarios (single legs, 2-leg, 3-leg props).
## Goals
### Predict NBA Player Prop Outcomes
Train machine learning models (e.g., XGBoost) to forecast player stat lines like Points, Assists, and Rebounds based on historical and contextual data.

### Build a Scalable Data Pipeline
- Create an automated pipeline to collect, clean, and store
- Historical player stats
- Opponent and team metrics
- Betting lines (PrizePicks, Underdog, sportsbooks)

### Calculate Smart Bets Using EV and proper risk management
Evaluate bets using:
- Expected Value (EV)
- Kelly Criterion

### Deploy a Streamlit Dashboard
- Display the best bets of the day, including:
- Predicted player stats from the model
- Over/Under recommendation
- EV rankings
- Parlay builder (1-leg, 2-leg, 3-leg props)

### How to use it
### To Scrape odds from prizepicks and odds api
```
python NBAPropFinder(region='us_dfs')
```
If you want the odds from below use region='us'
### Supported Sportsbooks from The Odds API
- FanDuel
- DraftKings
- BetMGM
- Caesars (William Hill)
- BetRivers
- PointsBet
- Bovada
- MyBookie.ag
- Unibet
- TwinSpires
- WynnBet
- LowVig.ag
- batPARX
- ESPN BET
- Fliff
- SI Sportsbook
- Tipico
- SuperBook
- Wind Creek (Betfred PA)
### To grab player and team stats
```
python getCompleteStats(season='2024-25', season_type='Regular Season')
```
## Example of what you get for a 2 leg w/ a $100 stake and a payout of $300 and odds at -137
<img width="1180" alt="Screenshot 2025-06-29 at 9 08 23 AM" src="https://github.com/user-attachments/assets/daa9366d-6d61-4f75-8a68-90100f576237" />

## Disclaimer
This project was created as a personal learning exercise and is intended for educational purposes only. The predictive models implemented here are experimental and have not demonstrated high accuracy in forecasting player performance. As such, the results should not be considered reliable for decision-making or betting purposes.
