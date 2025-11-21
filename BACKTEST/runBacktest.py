import pandas as pd
import multiprocessing as mp
from functools import partial
import joblib
import time
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from BACKTEST.backtest import backtestTrios

def process_date(date, df, backtestData, models, features):
    """Process one date - runs in parallel"""
    try:
        results = backtestTrios(
            data=df,
            backtestData=backtestData,
            gameDate=date,
            models=models,
            features=features,
            edge_threshold=0.20,
            top_n=15,
            variance_inflation=1.1,
            distribution_type='t',
            stat_col='PTS',
            use_monte_carlo=False,
            n_simulations=10000,
            max_kelly=0.25,
            stake=5
        )
        print(f"✓ {date}: {len(results)} bets")
        return results
    except Exception as e:
        print(f"✗ {date}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load everything
    print("Loading data and models...")
    # Load NGBoost mean and variance models
    mean_model = joblib.load(os.path.join(project_root, 'MODELS', 'SAVED_MODELS', 'NGBOOST_PTS_MEAN_MODEL_TEST.pkl'))
    variance_model = joblib.load(os.path.join(project_root, 'MODELS', 'SAVED_MODELS', 'NGBOOST_PTS_VAR_MODEL_TEST.pkl'))
    calibration_factor = joblib.load(os.path.join(project_root, 'MODELS', 'SAVED_MODELS', 'NGBOOST_PTS_CALIBRATION_FACTOR.pkl'))
    
    # Load isotonic regression calibrator (if available)
    iso_calibrator_path = os.path.join(project_root, 'MODELS', 'SAVED_MODELS', 'NGBOOST_PTS_ISOTONIC_CALIBRATOR_TEST.pkl')
    if os.path.exists(iso_calibrator_path):
        isotonic_calibrator = joblib.load(iso_calibrator_path)
        print("✓ Loaded isotonic regression calibrator")
    else:
        isotonic_calibrator = None
        print("⚠ Isotonic calibrator not found - using uncalibrated predictions")

    models = {
        'mean': mean_model,
        'variance': variance_model,
        'calibration_factor': calibration_factor,
        'isotonic_calibrator': isotonic_calibrator
    }

    features = joblib.load(os.path.join(project_root, 'MODELS', 'SAVED_MODELS', 'feature_list.pkl'))

    print(f"Loaded NGBoost models with calibration factor: {calibration_factor}")
    print(f"Using isotonic calibration: {isotonic_calibrator is not None}")
    print(f"Number of features: {len(features)}")

    # Load and prepare data
    s25_pts = pd.read_csv(os.path.join(project_root, 'DATA', 'CSV_FILES', 'TRAIN_DATA', 'PTS_TRAIN_25.csv'))
    s25_pts['IS_HIGH_SCORER'] = (s25_pts.groupby('PLAYER_ID')['PTS_AVG_TO_DATE'].transform('mean') > 18).astype(int)
    s24_pts = pd.read_csv(os.path.join(project_root, 'DATA', 'CSV_FILES', 'TRAIN_DATA', 'PTS_TRAIN_24.csv'))
    s24_pts['IS_HIGH_SCORER'] = (s24_pts.groupby('PLAYER_ID')['PTS_AVG_TO_DATE'].transform('mean') > 18).astype(int)

    df = pd.concat([s25_pts, s24_pts]).sort_values(by='GAME_DATE')

    backtestData = pd.read_csv(os.path.join(project_root, 'DATA', 'CSV_FILES', 'BACKTEST_DATA', 'dfs_data.csv'))
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
    if len(top_ev) > 0:
        print(f"Win Rate: {top_ev['parlay_won'].mean():.2%}")
    else:
        print("Win Rate: N/A (no bets)")

    # Calculate profit at $10 stake
    # 3-leg parlays typically pay 6x (so win = $60, lose = -$10)
    stake = 5
    wins = top_ev['parlay_won'].sum() if len(top_ev) > 0 else 0
    losses = len(top_ev) - wins
    total_staked = len(top_ev) * stake
    total_profit = (wins * 30) - (losses * stake)  # Fixed: 6x payout means $60 win for $10 stake

    print(f"\n${stake} per bet:")
    print(f"Staked: ${total_staked:,}")
    print(f"Profit: ${total_profit:,}")
    if total_staked > 0:
        print(f"ROI: {(total_profit/total_staked)*100:.2f}%")
    else:
        print("ROI: N/A")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'top_ev_recommended_trios.csv')
    top_ev.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

if __name__ == '__main__':
    main()