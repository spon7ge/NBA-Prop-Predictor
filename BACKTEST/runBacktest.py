import pandas as pd
import multiprocessing as mp
from functools import partial
import joblib
import time
from backtest import backtestTrios

def process_date(date, df, backtestData, models, features):
    """Process one date - runs in parallel"""
    try:
        results = backtestTrios(
            data=df,
            backtestData=backtestData,
            gameDate=date,
            models=models,
            features=features,
            edge_threshold=0.12,
            top_n=15,
            variance_inflation=1.1,
            distribution_type='t',
            stat_col='PTS',
            use_monte_carlo=False,
            n_simulations=10000,
            max_kelly=0.25,
            stake=100
        )
        print(f"✓ {date}: {len(results)} bets")
        return results
    except Exception as e:
        print(f"✗ {date}: {e}")
        return pd.DataFrame()

# Load everything
print("Loading data and models...")
# Load NGBoost mean and variance models
mean_model = joblib.load('../MODELS/SAVED_MODELS/NGBOOST_PTS_MEAN_MODEL.pkl')
variance_model = joblib.load('../MODELS/SAVED_MODELS/NGBOOST_PTS_VAR_MODEL.pkl')
calibration_factor = joblib.load('../MODELS/SAVED_MODELS/NGBOOST_PTS_CALIBRATION_FACTOR.pkl')

models = {
    'mean': mean_model,
    'variance': variance_model,
    'calibration_factor': calibration_factor
}

features = joblib.load('../MODELS/SAVED_MODELS/feature_list.pkl')

print(f"Loaded NGBoost models with calibration factor: {calibration_factor}")
print(f"Number of features: {len(features)}")

# Load and prepare data
s25_pts = pd.read_csv('../DATA/CSV_FILES/TRAIN_DATA/PTS_TRAIN_25.csv')
s25_pts['IS_HIGH_SCORER'] = (s25_pts.groupby('PLAYER_ID')['PTS_AVG_TO_DATE'].transform('mean') > 18).astype(int)
s24_pts = pd.read_csv('../DATA/CSV_FILES/TRAIN_DATA/PTS_TRAIN_24.csv')
s24_pts['IS_HIGH_SCORER'] = (s24_pts.groupby('PLAYER_ID')['PTS_AVG_TO_DATE'].transform('mean') > 18).astype(int)

df = pd.concat([s25_pts, s24_pts]).sort_values(by='GAME_DATE')

backtestData = pd.read_csv('../DATA/CSV_FILES/BACKTEST_DATA/dfs_data.csv')
backtestData = backtestData[
    (backtestData['BOOKMAKER'] == 'underdog') & 
    (backtestData['CATEGORY'] == 'player_points')
]

dates = sorted(backtestData['GAME_DATE'].unique())
print(f"Processing {len(dates)} dates\n")

# Run multiprocessing
start = time.time()

with mp.Pool(processes=10) as pool:
    process_func = partial(process_date, 
                          df=df, 
                          backtestData=backtestData, 
                          models=models, 
                          features=features)
    results = pool.map(process_func, dates)

# Combine results
valid_results = [r for r in results if not r.empty]
final_results = pd.concat(valid_results, ignore_index=True)

print(f"\n Done! {len(final_results)} total bets in {time.time()-start:.1f}s\n")

# Filter to Top EV recommended only
top_ev = final_results[
    (final_results['selection'] == 'top_ev') & 
    (final_results['parlay_recommendation'] == 1)
]

# Print results
print("="*60)
print("TOP EV RECOMMENDED TRIO BETS")
print("="*60)
print(f"Total Bets: {len(top_ev)}")
print(f"Win Rate: {top_ev['parlay_won'].mean():.2%}")

# Calculate profit at $5 stake
# 3-leg parlays typically pay 6x (so win = $30, lose = -$5)
stake = 5
wins = top_ev['parlay_won'].sum()
losses = len(top_ev) - wins
total_staked = len(top_ev) * stake
total_profit = (wins * 30) - (losses * stake)

print(f"\n${stake} per bet:")
print(f"Staked: ${total_staked:,}")
print(f"Profit: ${total_profit:,}")
print(f"ROI: {(total_profit/total_staked)*100:.2f}%")

# Save
top_ev.to_csv('top_ev_recommended_trios.csv', index=False)
print(f"\nSaved to: top_ev_recommended_trios.csv")