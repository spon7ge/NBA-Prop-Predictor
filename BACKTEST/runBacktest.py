import pandas as pd
import multiprocessing as mp
from functools import partial
import joblib
import time
from backtest import backtestPairs

def process_date(date, df, backtestData, models, features):
    """Process one date - runs in parallel"""
    try:
        results = backtestPairs(
            data=df,
            backtestData=backtestData,
            gameDate=date,
            models=models,
            features=features,
            edge_threshold=0.12,
            top_n=15,
            variance_inflation=1.1,
            distribution_type='skew_t',
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
models = {
    'q10': joblib.load('../MODELS/SAVED_MODELS/xgb_q10_modelv2.pkl'),
    'q50': joblib.load('../MODELS/SAVED_MODELS/xgb_q50_modelv2.pkl'),
    'q90': joblib.load('../MODELS/SAVED_MODELS/xgb_q90_modelv2.pkl')
}
features = joblib.load('../MODELS/SAVED_MODELS/feature_list.pkl')

df = pd.concat([
    pd.read_csv('../DATA/CSV_FILES/TRAIN_DATA/PTS_TRAIN_25.csv'),
    pd.read_csv('../DATA/CSV_FILES/TRAIN_DATA/PTS_TRAIN_24.csv')
]).sort_values(by='GAME_DATE')

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
    (final_results['pair_recommendation'] == 1)
]

# Print results
print("="*60)
print("TOP EV RECOMMENDED BETS")
print("="*60)
print(f"Total Bets: {len(top_ev)}")
print(f"Win Rate: {top_ev['pair_won'].mean():.2%}")

# Calculate profit at $5 stake
stake = 5
wins = top_ev['pair_won'].sum()
losses = len(top_ev) - wins
total_staked = len(top_ev) * stake
total_profit = (wins * 10) - (losses * stake)

print(f"\n${stake} per bet:")
print(f"Staked: ${total_staked:,}")
print(f"Profit: ${total_profit:,}")
print(f"ROI: {(total_profit/total_staked)*100:.2f}%")

# Save
top_ev.to_csv('top_ev_recommended.csv', index=False)
print(f"\nSaved to: top_ev_recommended.csv")