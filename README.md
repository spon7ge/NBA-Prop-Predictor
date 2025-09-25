# 🏀 NBA Player Prop Prediction
## Project Overview
Im building an end-to-end pipeline that predicts NBA player performance using features from players historical data and advanced stats to create ML models for points prop. Then grabbing the expected value (EV) and bet sizing using the Kelly Criterion for only single bets or 2 leg parlays
.
## Goals
### Predict NBA Player Prop Outcomes
Train machine a catboost model to forecast player stat lines like points based on historical and contextual data.

### Build a Scalable Data Pipeline
- Create an automated pipeline to collect, clean, and store
- Historical player stats
- Opponent and team metrics
- Betting lines (PrizePicks, Underdog, other US sportbooks)

### Calculate Smart Bets Using EV and proper risk management
Evaluate bets using:
- Monte Carlo Simulation to get the probability of it hitting 
- Expected Value (EV)
- Kelly Criterion

### Deploy a Streamlit Dashboard
- Display the best bets of the day, including:
- Predicted player stats from the model
- Over/Under recommendation
- EV rankings
- Parlay builder (1-leg, 2-leg)

If you want the odds from below use region='us'
### Supported Sportsbooks from The Odds API
- FanDuel *
- DraftKings *
- BetMGM *
- Caesars (William Hill) *
- BetRivers *
- PointsBet
- Bovada
- MyBookie.ag
- Unibet
- TwinSpires
- WynnBet
- LowVig.ag
- batPARX
- ESPN BET *
- Fliff
- SI Sportsbook
- Tipico
- SuperBook
- Wind Creek (Betfred PA)

## Example of what you get for a single bet
<img width="1326" height="303" alt="Screenshot 2025-09-25 at 4 37 12 PM" src="https://github.com/user-attachments/assets/62110db1-e23b-4c2b-9305-fb4017a0be5f" />

## Example of what you get for a 2-leg parlays
<img width="1397" height="272" alt="Screenshot 2025-09-25 at 4 38 53 PM" src="https://github.com/user-attachments/assets/17db00eb-902d-4120-9ad5-e8602324e930" />

## Disclaimer
This project was created as a personal learning exercise and is intended for educational purposes only. The predictive models implemented here are experimental and have not demonstrated high accuracy in forecasting player performance. As such, the results should not be considered reliable for decision-making or betting purposes.
