from datetime import datetime, timedelta
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, scheduleleaguev2, leaguedashteamstats
from src.utils.team_info import mainStartingFive, teamStarPlayer, projectedStartingFive, nameDict
from src.utils.helper_functions import findOpp, getUpcomingGamesCached

today = datetime.today().strftime('%Y-%m-%d')

league_df = leaguedashteamstats.LeagueDashTeamStats(
    league_id_nullable='00',
    per_mode_detailed='PerGame',
    measure_type_detailed_defense='Advanced'
).get_data_frames()[0]
if 'TEAM_ID' in league_df.columns:
    league_df = league_df.set_index('TEAM_ID')

_gameCache = {}
def get_league_stat(team_id, stat_name, default=100.0):
    try:
        if team_id in league_df.index:
            return league_df.at[team_id, stat_name]
        else:
            # Fallback: try to find by TEAM_ID column if index wasn't set properly
            if 'TEAM_ID' in league_df.columns:
                team_row = league_df[league_df['TEAM_ID'] == team_id]
                if not team_row.empty:
                    return team_row[stat_name].iloc[0]
            return default
    except (KeyError, IndexError):
        return default

# ----------------------------
# Utility helpers
# ----------------------------
def safe_mean(series):
    return float(series.mean()) if series.size > 0 else 0.0

def safe_std(series):
    return float(series.std()) if series.size > 0 else 0.0

def safe_delta(series, baseline):
    if series.size == 0:
        return 0.0
    return float(series.mean() - baseline)

# ----------------------------
# MIN Feature Builder
# ----------------------------
def player_min_features(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer, league_df):

    # ----------------------------
    # Locate player data
    # ----------------------------
    player_df = data[data["PLAYER_NAME"] == player_name].sort_values("GAME_DATE")
    if player_df.empty:
        print(f"[MIN] No data found for {player_name}")
        return None

    team = player_df["TEAM_ABBREVIATION"].iloc[-1]

    # ----------------------------
    # Opponent for the next game
    # ----------------------------
    opp, home_flag = findOpp(player_name, data, current_date)
    if opp is None:
        print(f"[MIN] No opponent found for {player_name}")
        return None

    opp_df = data[data["TEAM_ABBREVIATION"] == opp]
    if opp_df.empty:
        print(f"[MIN] Opponent data not found for {opp}")
        return None

    matchup_df = player_df[player_df["OPP_ABBREVIATION"] == opp]

    # ----------------------------
    # Team-level data
    # ----------------------------
    team_df = (
        data[data["TEAM_ABBREVIATION"] == team]
        .drop_duplicates("GAME_ID")
        .sort_values("GAME_DATE")
    )
    opp_team_df = (
        opp_df.drop_duplicates("GAME_ID")
        .sort_values("GAME_DATE")
    )

    # Calculate values needed for multiple features
    avg_min = safe_mean(player_df["MIN"])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    
    # Calculate days rest
    last_game = pd.to_datetime(player_df["GAME_DATE"]).max()
    player_days_rest = (pd.to_datetime(current_date) - last_game).days
    is_back_to_back = int(player_days_rest <= 1)
    
    # Calculate usual starters available
    main_starters = set(mainStartingFive.get(team, []))
    projected_starters = set(projectedStartingFive.get(team, []))
    usual_starters_available = len(main_starters - projected_starters)
    
    # Check if player missed last game
    if len(team_df) > 1 and len(player_df) > 0:
        last_team_game_date = pd.to_datetime(team_df["GAME_DATE"]).iloc[-1]
        last_player_game_date = pd.to_datetime(player_df["GAME_DATE"]).iloc[-1]
        player_missed_last = int(last_team_game_date > last_player_game_date)
    else:
        player_missed_last = 0
    
    # Calculate team min average
    team_min_avg = safe_mean(team_df["TEAM_MIN"]) if "TEAM_MIN" in team_df.columns else 240.0
    epsilon = 1e-8
    percentage_of_team_min = round(avg_min / (team_min_avg + epsilon), 4) if team_min_avg > 0 else 0.0
    
    # Calculate MIN team rank
    team_players_df = data[data["TEAM_ABBREVIATION"] == team].copy()
    if not team_players_df.empty:
        team_min_avgs = {}
        for team_player_name in team_players_df["PLAYER_NAME"].unique():
            team_player_df = data[data["PLAYER_NAME"] == team_player_name].sort_values("GAME_DATE")
            if not team_player_df.empty:
                team_min_avgs[team_player_name] = safe_mean(team_player_df["MIN"])
        
        if team_min_avgs:
            min_series = pd.Series(team_min_avgs)
            min_ranks = min_series.rank(method='dense', ascending=False)
            min_team_rank = float(min_ranks.get(player_name, len(team_min_avgs) + 1))
        else:
            min_team_rank = 1.0
    else:
        min_team_rank = 1.0
    
    # Calculate min last 5 stats
    min_last_5 = player_df["MIN"].tail(5)
    min_ceiling_l5 = float(min_last_5.max()) if min_last_5.size > 0 else 0.0
    min_floor_l5 = float(min_last_5.min()) if min_last_5.size > 0 else 0.0
    
    # Calculate star boost
    star_missing = int(teamStarPlayer.get(team, None) not in projectedStartingFive.get(team, []))
    starOut = player_df[player_df.get("STAR_SAT_OUT", pd.Series([0])) == 1]
    starIn = player_df[player_df.get("STAR_SAT_OUT", pd.Series([0])) == 0]
    star_boost = safe_mean(starOut["MIN"]) - safe_mean(starIn["MIN"])
    
    # Calculate league and team averages
    league_pace_avg = safe_mean(league_df["PACE"]) if "PACE" in league_df.columns else 100.0
    league_def_avg = safe_mean(league_df["DEF_RATING"]) if "DEF_RATING" in league_df.columns else 110.0
    team_pace = safe_mean(team_df["TEAM_PACE"])
    opp_pace = safe_mean(opp_team_df["TEAM_PACE"])
    opp_def_avg = safe_mean(opp_team_df["TEAM_DEF_RATING"])

    # Build features list in the EXACT order from MIN_features
    res = []
    
    # 1: STARTING_X_MIN
    res.append(round(starting_flag * avg_min, 2))
    
    # 2: PLAYER_DAYS_REST
    res.append(player_days_rest)
    
    # 3: USUAL_STARTERS_AVAILABLE
    res.append(usual_starters_available)
    
    # 4: B2B_X_MIN
    res.append(round(is_back_to_back * avg_min, 2))
    
    # 5: PLAYER_MISSED_LAST_GAME_X_MIN
    res.append(round(player_missed_last * avg_min, 2))
    
    # 6: PERCENTAGE_OF_TEAM_MIN
    res.append(percentage_of_team_min)
    
    # 7: MIN_TEAM_RANK
    res.append(min_team_rank)
    
    # 8: MIN_CEILING_L5_DELTA
    res.append(round(min_ceiling_l5 - avg_min, 2) if min_last_5.size > 0 else 0.0)
    
    # 9: MIN_FLOOR_L5_DELTA
    res.append(round(min_floor_l5 - avg_min, 2) if min_last_5.size > 0 else 0.0)
    
    # 10: MIN_AVG_TO_DATE
    res.append(avg_min)
    
    # 11: MIN_L5_OVER_BASELINE
    res.append(safe_delta(min_last_5, avg_min))
    
    # 12: MIN_L10_OVER_BASELINE
    res.append(safe_delta(player_df["MIN"].tail(10), avg_min))
    
    # 13: MIN_STD_10_TO_DATE
    res.append(safe_std(player_df["MIN"].tail(10)))
    
    # 14: MIN_BOOST_STAR_OUT
    res.append(star_missing * star_boost)
    
    # 15: MATCHUP_MIN_DELTA
    res.append(safe_delta(matchup_df["MIN"], avg_min))
    
    # 16: TEAM_PACE_OVER_LEAGUE_AVG
    res.append(team_pace - league_pace_avg)
    
    # 17: EXPECTED_PACE
    res.append((team_pace + opp_pace) / 2)
    
    # 18: OPP_DEF_RATING_OVER_LEAGUE_AVG
    res.append(opp_def_avg - league_def_avg)
    
    # 19: OPP_PACE_OVER_LEAGUE_AVG
    res.append(opp_pace - league_pace_avg)

    return res


# ----------------------------
# Wrapper used by inference
# ----------------------------
def buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer, league_df):
    features = player_min_features(
        player_name, data, current_date,
        projectedStartingFive, mainStartingFive,
        teamStarPlayer, league_df
    )

    if features is None:
        print(f"[MIN] Failed to generate features for {player_name}")
        return None

    return [features]