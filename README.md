# 🏀 NBA Player Prop Prediction
## Project Overview
Im building an end-to-end pipeline that predicts NBA player performance using features from players historical data and advanced stats to create a catboost/xgboost model for points. Then grabbing the expected value (EV) and bet sizing using the Kelly Criterion for only single bets or 2 leg parlays to find the best bets.
.
## Goals
### Predict NBA Player Prop Outcomes
Training a catboost and xgboost model to predict a players stat line like points based on historical and contextual data.

### Build a Scalable Data Pipeline
- Create an automated pipeline to collect, clean, and store
- Historical player stats
- Opponent and team metrics
- Feature Engineering
- Betting lines (PrizePicks, Underdog, other US sportbooks)

### Calculate Smart Bets Using EV and proper risk management
Evaluate bets using:
- Monte Carlo Simulation 
- Expected Value (EV)
- Kelly Criterion

### Deploy a Streamlit Dashboard (End Goal)
- Display the top 10 EVs for the day:
- Predicted player stats from the model
- Over/Under recommendation
- EV rankings
- Parlay builder (1-leg, 2-leg, 3-leg)

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

## Current problems and changes coming:
- grabing team spreads and total per bookmakers and assigning the average to the current team
- assigning the current opponent when I run the predict function
- changing player positions from G,F,C to PG,SG,SF,PF,C

## Example of what you get for a single bet only using bookmakers that allow single bets
<img width="1326" height="303" alt="Screenshot 2025-09-25 at 4 37 12 PM" src="https://github.com/user-attachments/assets/62110db1-e23b-4c2b-9305-fb4017a0be5f" />

## Example of what you get for a 2-leg parlay only using DFS bookmakers (The lines below are from underdog)
<img width="1397" height="272" alt="Screenshot 2025-09-25 at 4 38 53 PM" src="https://github.com/user-attachments/assets/17db00eb-902d-4120-9ad5-e8602324e930" />

## Disclaimer
This project was created as a personal learning exercise and is intended for educational purposes only. The predictive models implemented here are experimental and have not demonstrated high accuracy in forecasting player performance. As such, the results should not be considered reliable for decision-making or betting purposes.
