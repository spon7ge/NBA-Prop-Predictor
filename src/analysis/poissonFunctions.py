import pandas as pd
import numpy as np


def _get_rest_factor(days_rested):
    if days_rested == 1:
        return 0.93
    elif days_rested == 2:
        return 1.00
    elif 3 <= days_rested <= 5:
        return 1.02
    else:
        return 0.98


def _get_days_rested(player_df, current_date):
    current_date_dt = pd.to_datetime(current_date)
    player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
    last_game_date = player_df['GAME_DATE'].max()
    return (current_date_dt - last_game_date).days


def _get_h2h_factor(player_df, player_df_25, opp_team, stat_col, baseline_mean, min_factor=0.85, max_factor=1.15):
    h2h = player_df[player_df['OPP_ABBREVIATION'] == opp_team]
    if h2h.empty and not player_df_25.empty:
        h2h = player_df_25[player_df_25['OPP_ABBREVIATION'] == opp_team]
    if h2h.empty:
        return 1.0
    h2h_avg = h2h[stat_col].mean()
    h2h_factor = h2h_avg / baseline_mean if baseline_mean > 0 else 1.0
    return max(min_factor, min(max_factor, h2h_factor))


def _get_starting_factor(player_name, player_team_abbr, projectedStartingFive, starting_boost=1.12, bench_reduction=0.85):
    is_starting = player_name in projectedStartingFive.get(player_team_abbr, [])
    return starting_boost if is_starting else bench_reduction


def _compute_posterior(player_df, stat_col, prior_alpha, prior_beta, recent_games=7):
    recent_games = min(recent_games, len(player_df))
    recent_stats = player_df[stat_col].tail(recent_games).values
    posterior_alpha = prior_alpha + recent_stats.sum()
    posterior_beta = prior_beta + recent_games
    lambda_adjusted = posterior_alpha / posterior_beta
    posterior_variance = posterior_alpha / (posterior_beta ** 2)
    posterior_std = np.sqrt(posterior_variance)
    return lambda_adjusted, posterior_std


def compute_bayesian_lambda(player_df, player_df_25, player_team, player_team_abbr, opp_team_id, opp_team,
                           team_stats, league_avg_off_rtg, league_avg_def_rtg,
                           league_avg_pace, home_flag, current_date, player_name, projectedStartingFive):
    if len(player_df) < 5:
        baseline_mean = player_df_25['PTS'].mean() if not player_df_25.empty else 10.0
        baseline_games = len(player_df_25)
    else:
        baseline_mean = player_df['PTS'].mean()
        baseline_games = len(player_df)
    
    if baseline_mean <= 0:
        return None
    
    team_or = team_stats.at[player_team, 'OFF_RATING']
    team_pace = team_stats.at[player_team, 'PACE']
    opp_dr = team_stats.at[opp_team_id, 'DEF_RATING']
    opp_pace = team_stats.at[opp_team_id, 'PACE']
    
    team_or_factor = team_or / league_avg_off_rtg
    opp_dr_factor = league_avg_def_rtg / opp_dr
    expected_pace = (team_pace + opp_pace) / 2
    pace_factor = expected_pace / league_avg_pace
    home_factor = 1.03 if home_flag else 0.97
    
    days_rested = _get_days_rested(player_df, current_date)
    rest_factor = _get_rest_factor(days_rested)
    starting_factor = _get_starting_factor(player_name, player_team_abbr, projectedStartingFive, 1.12, 0.85)
    h2h_factor = _get_h2h_factor(player_df, player_df_25, opp_team, 'PTS', baseline_mean)
    
    prior_mean = baseline_mean * team_or_factor * opp_dr_factor * pace_factor * home_factor * rest_factor * h2h_factor * starting_factor
    prior_strength = max(5, min(baseline_games, 40))
    prior_beta = prior_strength / 10.0
    prior_alpha = prior_mean * prior_beta
    
    if prior_alpha < 1:
        prior_alpha = 1.0
        prior_beta = prior_alpha / prior_mean
    
    return _compute_posterior(player_df, 'PTS', prior_alpha, prior_beta, 7)


def compute_bayesian_lambda_assists(player_df, player_df_25, player_team, player_team_abbr, opp_team_id, opp_team,
                                   team_stats, league_avg_def_rtg, league_avg_pace,
                                   league_avg_ast_ratio, league_avg_tov, home_flag, current_date, player_name, projectedStartingFive):
    if len(player_df) < 5:
        baseline_mean = player_df_25['AST'].mean() if not player_df_25.empty else 3.0
        baseline_games = len(player_df_25)
    else:
        baseline_mean = player_df['AST'].mean()
        baseline_games = len(player_df)
    
    if baseline_mean <= 0:
        return None
    
    team_pace = team_stats.at[player_team, 'PACE']
    team_ast_ratio = team_stats.at[player_team, 'AST_RATIO']
    opp_dr = team_stats.at[opp_team_id, 'DEF_RATING']
    opp_pace = team_stats.at[opp_team_id, 'PACE']
    opp_tov = team_stats.at[opp_team_id, 'TM_TOV_PCT']
    
    team_ast_ratio_factor = team_ast_ratio / league_avg_ast_ratio
    opp_dr_factor = league_avg_def_rtg / opp_dr
    opp_tov_factor = league_avg_tov / opp_tov
    expected_pace = (team_pace + opp_pace) / 2
    pace_factor = expected_pace / league_avg_pace
    home_factor = 1.02 if home_flag else 0.98
    
    days_rested = _get_days_rested(player_df, current_date)
    rest_factor = _get_rest_factor(days_rested)
    starting_factor = _get_starting_factor(player_name, player_team_abbr, projectedStartingFive, 1.10, 0.88)
    h2h_factor = _get_h2h_factor(player_df, player_df_25, opp_team, 'AST', baseline_mean)
    
    prior_mean = baseline_mean * team_ast_ratio_factor * opp_dr_factor * opp_tov_factor * pace_factor * home_factor * rest_factor * h2h_factor * starting_factor
    prior_strength = max(5, min(baseline_games, 40))
    prior_beta = prior_strength / 10.0
    prior_alpha = prior_mean * prior_beta
    
    if prior_alpha < 1:
        prior_alpha = 1.0
        prior_beta = prior_alpha / prior_mean
    
    return _compute_posterior(player_df, 'AST', prior_alpha, prior_beta, 7)


def compute_bayesian_lambda_rebounds(player_df, player_df_25, player_team, player_team_abbr, opp_team_id, opp_team,
                                     team_stats, league_avg_pace, league_avg_reb,
                                     league_avg_oreb, league_avg_dreb, home_flag, current_date, player_name, projectedStartingFive):
    if len(player_df) < 5:
        baseline_mean = player_df_25['REB'].mean() if not player_df_25.empty else 5.0
        baseline_games = len(player_df_25)
    else:
        baseline_mean = player_df['REB'].mean()
        baseline_games = len(player_df)
    
    if baseline_mean <= 0:
        return None
    
    team_pace = team_stats.at[player_team, 'PACE']
    team_reb = team_stats.at[player_team, 'REB_PCT']
    opp_pace = team_stats.at[opp_team_id, 'PACE']
    opp_oreb = team_stats.at[opp_team_id, 'OREB_PCT']
    opp_dreb = team_stats.at[opp_team_id, 'DREB_PCT']
    
    team_reb_factor = team_reb / league_avg_reb
    opp_dreb_factor = league_avg_dreb / opp_dreb
    opp_oreb_factor = league_avg_oreb / opp_oreb
    expected_pace = (team_pace + opp_pace) / 2
    pace_factor = expected_pace / league_avg_pace
    home_factor = 1.01 if home_flag else 0.99
    
    days_rested = _get_days_rested(player_df, current_date)
    rest_factor = _get_rest_factor(days_rested)
    starting_factor = _get_starting_factor(player_name, player_team_abbr, projectedStartingFive, 1.10, 0.88)
    h2h_factor = _get_h2h_factor(player_df, player_df_25, opp_team, 'REB', baseline_mean)
    
    prior_mean = baseline_mean * team_reb_factor * opp_dreb_factor * opp_oreb_factor * pace_factor * home_factor * rest_factor * h2h_factor * starting_factor
    prior_strength = max(5, min(baseline_games, 40))
    prior_beta = prior_strength / 10.0
    prior_alpha = prior_mean * prior_beta
    
    if prior_alpha < 1:
        prior_alpha = 1.0
        prior_beta = prior_alpha / prior_mean
    
    return _compute_posterior(player_df, 'REB', prior_alpha, prior_beta, 7)


def compute_bayesian_lambda_blocks(player_df, player_df_25, player_team, player_team_abbr, opp_team_id, opp_team,
                                   team_stats, league_avg_pace, home_flag, current_date, player_name, projectedStartingFive):
    if len(player_df) < 5:
        baseline_mean = player_df_25['BLK'].mean() if not player_df_25.empty else 0.5
        baseline_games = len(player_df_25)
    else:
        baseline_mean = player_df['BLK'].mean()
        baseline_games = len(player_df)
    
    if baseline_mean <= 0:
        baseline_mean = 0.1
    
    team_pace = team_stats.at[player_team, 'PACE']
    opp_pace = team_stats.at[opp_team_id, 'PACE']
    
    expected_pace = (team_pace + opp_pace) / 2
    pace_factor = expected_pace / league_avg_pace
    home_factor = 1.00
    
    days_rested = _get_days_rested(player_df, current_date)
    rest_factor = _get_rest_factor(days_rested)
    starting_factor = _get_starting_factor(player_name, player_team_abbr, projectedStartingFive, 1.15, 0.80)
    h2h_factor = _get_h2h_factor(player_df, player_df_25, opp_team, 'BLK', baseline_mean, 0.80, 1.20)
    
    prior_mean = baseline_mean * pace_factor * home_factor * rest_factor * h2h_factor * starting_factor
    prior_strength = max(3, min(baseline_games, 30))
    prior_beta = prior_strength / 8.0
    prior_alpha = prior_mean * prior_beta
    
    if prior_alpha < 0.1:
        prior_alpha = 0.1
        prior_beta = prior_alpha / prior_mean
    
    lambda_adjusted, posterior_std = _compute_posterior(player_df, 'BLK', prior_alpha, prior_beta, 10)
    
    if lambda_adjusted <= 0:
        lambda_adjusted = 0.1
    
    return lambda_adjusted, posterior_std


def compute_bayesian_lambda_steals(player_df, player_df_25, player_team, player_team_abbr, opp_team_id, opp_team,
                                   team_stats, league_avg_pace, league_avg_tov,
                                   home_flag, current_date, player_name, projectedStartingFive):
    if len(player_df) < 5:
        baseline_mean = player_df_25['STL'].mean() if not player_df_25.empty else 0.8
        baseline_games = len(player_df_25)
    else:
        baseline_mean = player_df['STL'].mean()
        baseline_games = len(player_df)
    
    if baseline_mean <= 0:
        baseline_mean = 0.1
    
    team_pace = team_stats.at[player_team, 'PACE']
    opp_pace = team_stats.at[opp_team_id, 'PACE']
    opp_tov = team_stats.at[opp_team_id, 'TM_TOV_PCT']
    
    opp_tov_factor = opp_tov / league_avg_tov
    expected_pace = (team_pace + opp_pace) / 2
    pace_factor = expected_pace / league_avg_pace
    home_factor = 1.00
    
    days_rested = _get_days_rested(player_df, current_date)
    rest_factor = _get_rest_factor(days_rested)
    starting_factor = _get_starting_factor(player_name, player_team_abbr, projectedStartingFive, 1.12, 0.82)
    h2h_factor = _get_h2h_factor(player_df, player_df_25, opp_team, 'STL', baseline_mean, 0.80, 1.20)
    
    prior_mean = baseline_mean * opp_tov_factor * pace_factor * home_factor * rest_factor * h2h_factor * starting_factor
    prior_strength = max(3, min(baseline_games, 30))
    prior_beta = prior_strength / 8.0
    prior_alpha = prior_mean * prior_beta
    
    if prior_alpha < 0.1:
        prior_alpha = 0.1
        prior_beta = prior_alpha / prior_mean
    
    lambda_adjusted, posterior_std = _compute_posterior(player_df, 'STL', prior_alpha, prior_beta, 10)
    
    if lambda_adjusted <= 0:
        lambda_adjusted = 0.1
    
    return lambda_adjusted, posterior_std

