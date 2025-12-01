import pandas as pd
import numpy as np
from datetime import datetime
from nba_api.stats.endpoints import leaguedashteamstats

def compute_bayesian_lambda_points_v2(
    player_df, 
    player_df_prev_season,
    player_team_id,
    player_team_abbr,
    opp_team_id,
    opp_team_abbr,
    team_stats,
    league_avg_off_rtg,
    league_avg_def_rtg,
    league_avg_pace,
    home_flag,
    current_date,
    player_name,
    projectedStartingFive,
    predicted_minutes=None,
    predicted_usage=None
):
    """
    Enhanced Bayesian Poisson with better handling of predicted MIN/USG.
    """
    epsilon = 1e-6
    
    # Get baseline from season data
    if len(player_df) < 5:
        baseline_mean = player_df_prev_season['PTS'].mean() if not player_df_prev_season.empty else 10.0
        baseline_games = len(player_df_prev_season)
        baseline_std = player_df_prev_season['PTS'].std() if not player_df_prev_season.empty else baseline_mean * 0.3
    else:
        baseline_mean = player_df['PTS'].mean()
        baseline_games = len(player_df)
        baseline_std = player_df['PTS'].std()
    
    if baseline_mean <= 0:
        return None
    
    # Get team and opponent stats
    try:
        team_or = team_stats.at[player_team_id, 'OFF_RATING']
        team_pace = team_stats.at[player_team_id, 'PACE']
        opp_dr = team_stats.at[opp_team_id, 'DEF_RATING']
        opp_pace = team_stats.at[opp_team_id, 'PACE']
    except (KeyError, IndexError):
        return None
    
    # ========================================================================
    # CONTEXTUAL ADJUSTMENTS (in log space for stability)
    # ========================================================================
    
    log_adjustments = {}
    
    # Team offensive strength
    log_adjustments['team_offense'] = np.log(team_or / league_avg_off_rtg)
    
    # Opponent defensive weakness
    log_adjustments['opp_defense'] = np.log(league_avg_def_rtg / opp_dr)
    
    # Pace adjustment
    expected_pace = (team_pace + opp_pace) / 2
    log_adjustments['pace'] = np.log(expected_pace / league_avg_pace)
    
    # Home court advantage
    log_adjustments['home'] = np.log(1.03 if home_flag else 0.97)
    
    # Days rested
    current_date_dt = pd.to_datetime(current_date)
    player_df_copy = player_df.copy()
    player_df_copy['GAME_DATE'] = pd.to_datetime(player_df_copy['GAME_DATE'])
    last_game_date = player_df_copy['GAME_DATE'].max()
    days_rested = (current_date_dt - last_game_date).days
    
    if days_rested == 1:
        rest_factor = 0.93
    elif days_rested == 2:
        rest_factor = 1.00
    elif 3 <= days_rested <= 5:
        rest_factor = 1.02
    else:
        rest_factor = 0.98
    log_adjustments['rest'] = np.log(rest_factor)
    
    # Starting status
    is_starting = player_name in projectedStartingFive.get(player_team_abbr, [])
    starting_factor = 1.12 if is_starting else 0.85
    log_adjustments['starting'] = np.log(starting_factor)
    
    # Head-to-head
    h2h = player_df[player_df['OPP_ABBREVIATION'] == opp_team_abbr]
    if h2h.empty and not player_df_prev_season.empty:
        h2h = player_df_prev_season[player_df_prev_season['OPP_ABBREVIATION'] == opp_team_abbr]
    
    if not h2h.empty:
        h2h_avg = h2h['PTS'].mean()
        h2h_factor = h2h_avg / baseline_mean if baseline_mean > 0 else 1.0
        h2h_factor = max(0.85, min(1.15, h2h_factor))
        log_adjustments['h2h'] = np.log(h2h_factor)
    
    # Combine adjustments
    total_log_adjustment = sum(log_adjustments.values())
    context_factor = np.exp(total_log_adjustment)
    
    # Apply contextual adjustments
    prior_mean = baseline_mean * context_factor
    
    # ========================================================================
    # INCORPORATE PREDICTED MINUTES AND USAGE (Key improvement!)
    # ========================================================================
    
    if predicted_minutes is not None and predicted_usage is not None:
        # Get historical averages
        avg_minutes = player_df['MIN'].mean() if len(player_df) > 0 else 30.0
        avg_usage = player_df['USG_PCT'].mean() if len(player_df) > 0 else 0.20
        
        # Calculate expected possessions
        # Possessions = (Minutes / 48) * Team Pace * 0.96 (adjustment factor)
        expected_team_poss = expected_pace * 0.96
        
        # Player's share of possessions
        expected_player_poss = (predicted_minutes / 48.0) * expected_team_poss * predicted_usage
        avg_player_poss = (avg_minutes / 48.0) * expected_team_poss * avg_usage
        
        # Possession-based adjustment
        if avg_player_poss > 0:
            poss_factor = expected_player_poss / avg_player_poss
            # More generous bounds (allow 2.5x on high end for blowup games)
            poss_factor = max(0.3, min(2.5, poss_factor))
        else:
            poss_factor = 1.0
        
        # Apply possession adjustment
        prior_mean = prior_mean * poss_factor
    
    # Bounds check on final prior mean
    prior_mean = max(0.5, min(prior_mean, baseline_mean * 3.0))
    
    # ========================================================================
    # PRIOR STRENGTH (accounts for consistency)
    # ========================================================================
    
    pts_cv = baseline_std / baseline_mean if baseline_mean > 0 else 0.3
    consistency_factor = 1.0 / (1.0 + pts_cv)
    
    effective_games = baseline_games * (0.5 + 0.5 * consistency_factor)
    prior_strength = min(effective_games, 60)
    prior_strength = max(prior_strength, 3)
    
    # Gamma prior parameters
    prior_beta = prior_strength / 10.0
    prior_alpha = prior_mean * prior_beta
    
    # Ensure valid Gamma
    if prior_alpha < 1:
        prior_alpha = 1.0
        prior_beta = prior_alpha / prior_mean
    
    # ========================================================================
    # BAYESIAN UPDATE with weighted recent games
    # ========================================================================
    
    recent_games = min(15, len(player_df))
    recent_pts = player_df['PTS'].tail(recent_games).values
    
    # Exponential weighting (more recent = more important)
    weights = np.exp(np.linspace(-1, 0, len(recent_pts)))
    weights_normalized = weights / weights.sum()
    
    # Weighted contribution to posterior
    weighted_sum = np.sum(recent_pts * weights_normalized * len(recent_pts))
    
    # Posterior update
    posterior_alpha = prior_alpha + weighted_sum
    posterior_beta = prior_beta + recent_games
    
    # Posterior mean (adjusted lambda)
    lambda_adjusted = posterior_alpha / posterior_beta
    
    # Posterior uncertainty
    posterior_variance = posterior_alpha / (posterior_beta ** 2)
    posterior_std = np.sqrt(posterior_variance)
    
    return lambda_adjusted, posterior_std


def predict_points_poisson(
    player_name,
    data,
    date,
    projectedStartingFive=None,
    mainStartingFive=None,
    teamStarPlayer=None,
    league_df=None,
    findOpp=None,
    prev_season_data=None,
    predicted_minutes=None,
    predicted_usage=None
):
    """
    Predict points using Poisson model with Bayesian lambda.
    
    Returns:
        dict: {
            'predicted_points': float,
            'lambda': float,
            'sigma': float
        } or None if prediction fails
    """
    # Import dependencies if not provided
    if projectedStartingFive is None:
        from PRODUCTION.teamInfo import projectedStartingFive
    
    if findOpp is None:
        from PRODUCTION.pipelineV2 import findOpp
    
    if league_df is None:
        league_df = leaguedashteamstats.LeagueDashTeamStats(
            league_id_nullable='00',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Advanced'
        ).get_data_frames()[0]
        if 'TEAM_ID' in league_df.columns:
            league_df = league_df.set_index('TEAM_ID')
    
    # Get player data
    player_df = data[data['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')
    if player_df.empty:
        return None
    
    # Get team info
    player_team_id = player_df['TEAM_ID'].iloc[-1]
    player_team_abbr = player_df['TEAM_ABBREVIATION'].iloc[-1]
    
    # Get opponent
    opp_team_abbr, home_flag = findOpp(player_name, data, date)
    if opp_team_abbr is None:
        return None
    
    # Get opponent team data
    opp_matches = data[data['TEAM_ABBREVIATION'] == opp_team_abbr]
    if opp_matches.empty:
        return None
    opp_team_id = opp_matches['TEAM_ID'].iloc[-1]
    
    # Get previous season data if available
    if prev_season_data is None:
        player_df_prev = pd.DataFrame()
    else:
        player_df_prev = prev_season_data[prev_season_data['PLAYER_NAME'] == player_name]
    
    # Get league stats - ensure TEAM_ID is the index for lookups
    if isinstance(league_df, pd.DataFrame):
        # Check if TEAM_ID is already the index
        if league_df.index.name == 'TEAM_ID':
            team_stats = league_df
        elif 'TEAM_ID' in league_df.columns:
            # Set TEAM_ID as index if it's a column
            team_stats = league_df.set_index('TEAM_ID')
        else:
            # If TEAM_ID is not in columns or index, use as-is (might cause issues)
            team_stats = league_df
    else:
        team_stats = league_df
    
    league_avg_off_rtg = team_stats['OFF_RATING'].mean()
    league_avg_def_rtg = team_stats['DEF_RATING'].mean()
    league_avg_pace = team_stats['PACE'].mean()
    
    # Compute Bayesian lambda using v2 function
    result = compute_bayesian_lambda_points_v2(
        player_df,
        player_df_prev,
        player_team_id,
        player_team_abbr,
        opp_team_id,
        opp_team_abbr,
        team_stats,
        league_avg_off_rtg,
        league_avg_def_rtg,
        league_avg_pace,
        home_flag,
        date,
        player_name,
        projectedStartingFive,
        predicted_minutes=predicted_minutes,
        predicted_usage=predicted_usage
    )
    
    if result is None:
        return None
    
    lambda_adjusted, posterior_std = result
    
    return {
        'predicted_points': float(lambda_adjusted),
        'lambda': float(lambda_adjusted),
        'sigma': float(posterior_std)
    }