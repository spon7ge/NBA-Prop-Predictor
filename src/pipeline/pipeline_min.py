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

    res = []

    # ============================
    # Calculate avg_min early (needed for multiple features)
    # ============================
    avg_min = safe_mean(player_df["MIN"])

    # ============================
    # 1 — STARTING_X_MIN
    # ============================
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    starting_x_min = round(starting_flag * avg_min, 2)
    res.append(starting_x_min)

    # ============================
    # 2 — STAR_SAT_OUT
    # ============================
    star_sat_out = int(teamStarPlayer.get(team, None) not in projectedStartingFive.get(team, []))
    res.append(star_sat_out)

    # ============================
    # 3 — PLAYER_DAYS_REST
    # ============================
    last_game = pd.to_datetime(player_df["GAME_DATE"]).max()
    player_days_rest = (pd.to_datetime(current_date) - last_game).days
    res.append(player_days_rest)

    # ============================
    # 4 — USUAL_STARTERS_AVAILABLE
    # ============================
    main_starters = set(mainStartingFive.get(team, []))
    projected_starters = set(projectedStartingFive.get(team, []))
    usual_starters_available = len(main_starters - projected_starters)
    res.append(usual_starters_available)

    # ============================
    # 5 — B2B_X_MIN (Back to Back interaction with MIN)
    # ============================
    is_back_to_back = int(player_days_rest <= 1)
    b2b_x_min = round(is_back_to_back * avg_min, 2)
    res.append(b2b_x_min)

    # ============================
    # 6 — PLAYER_MISSED_LAST_GAME_X_MIN
    # ============================
    # Check if player missed the last team game
    # Logic: if there was a previous team game and the player didn't play in it
    if len(team_df) > 1 and len(player_df) > 0:
        # Get the most recent team game
        last_team_game_date = pd.to_datetime(team_df["GAME_DATE"]).iloc[-1]
        # Get the most recent player game
        last_player_game_date = pd.to_datetime(player_df["GAME_DATE"]).iloc[-1]
        # Check if the last team game exists and player didn't play in it
        player_missed_last = int(last_team_game_date > last_player_game_date)
    else:
        player_missed_last = 0
    player_missed_last_x_min = round(player_missed_last * avg_min, 2)
    res.append(player_missed_last_x_min)

    # ============================
    # 7 — PERCENTAGE_OF_TEAM_MIN
    # ============================
    # Calculate team total minutes (average team minutes per game)
    team_min_avg = safe_mean(team_df["TEAM_MIN"]) if "TEAM_MIN" in team_df.columns else 240.0
    epsilon = 1e-8
    percentage_of_team_min = round(avg_min / (team_min_avg + epsilon), 4) if team_min_avg > 0 else 0.0
    res.append(percentage_of_team_min)

    # ============================
    # 8 — MIN_TEAM_RANK
    # ============================
    # Calculate MIN rank among all players on the team
    # Get all players on the team with their MIN averages
    team_players_df = data[data["TEAM_ABBREVIATION"] == team].copy()
    if not team_players_df.empty:
        # Calculate MIN_AVG_TO_DATE for each player on the team
        team_min_avgs = {}
        for team_player_name in team_players_df["PLAYER_NAME"].unique():
            team_player_df = data[data["PLAYER_NAME"] == team_player_name].sort_values("GAME_DATE")
            if not team_player_df.empty:
                team_min_avgs[team_player_name] = safe_mean(team_player_df["MIN"])
        
        # Create a series and rank (ascending=False means rank 1 = highest MIN)
        if team_min_avgs:
            min_series = pd.Series(team_min_avgs)
            min_ranks = min_series.rank(method='dense', ascending=False)
            min_team_rank = float(min_ranks.get(player_name, len(team_min_avgs) + 1))
        else:
            min_team_rank = 1.0
    else:
        min_team_rank = 1.0
    res.append(min_team_rank)

    # ============================
    # 9 — MIN_CEILING_L5_DELTA
    # ============================
    min_last_5 = player_df["MIN"].tail(5)
    min_ceiling_l5 = float(min_last_5.max()) if min_last_5.size > 0 else 0.0
    min_ceiling_l5_delta = round(min_ceiling_l5 - avg_min, 2) if min_last_5.size > 0 else 0.0
    res.append(min_ceiling_l5_delta)

    # ============================
    # 10 — MIN_FLOOR_L5_DELTA
    # ============================
    min_floor_l5 = float(min_last_5.min()) if min_last_5.size > 0 else 0.0
    min_floor_l5_delta = round(min_floor_l5 - avg_min, 2) if min_last_5.size > 0 else 0.0
    res.append(min_floor_l5_delta)

    # ============================
    # 11 — STARTING_x_MIN_AVG
    # ============================
    starting_x_min_avg = round(starting_flag * avg_min, 2)
    res.append(starting_x_min_avg)

    # ============================
    # 12 — STARTING_x_MIN_CEILING
    # ============================
    starting_x_min_ceiling = round(starting_flag * min_ceiling_l5, 2)
    res.append(starting_x_min_ceiling)

    # ============================
    # 13 — HIGH_MIN_PLAYER
    # ============================
    high_min_player = int(avg_min >= 25)
    res.append(high_min_player)

    # ============================
    # 14 — MEDIUM_MIN_PLAYER
    # ============================
    medium_min_player = int(avg_min >= 12 and avg_min < 25)
    res.append(medium_min_player)

    # ============================
    # 15 — LOW_MIN_PLAYER
    # ============================
    low_min_player = int(avg_min < 12)
    res.append(low_min_player)

    # ============================
    # 16 — MIN_AVG_TO_DATE
    # ============================
    res.append(avg_min)

    # ============================
    # 17 — MIN_L5_OVER_BASELINE
    # ============================
    res.append(safe_delta(player_df["MIN"].tail(5), avg_min))

    # ============================
    # 18 — MIN_STD_5_TO_DATE
    # ============================
    res.append(safe_std(player_df["MIN"].tail(5)))

    # ============================
    # 19 — MIN_BOOST_STAR_OUT
    # ============================
    star_missing = int(teamStarPlayer.get(team, None) not in projectedStartingFive.get(team, []))
    starOut = player_df[player_df.get("STAR_SAT_OUT", pd.Series([0])) == 1]
    starIn = player_df[player_df.get("STAR_SAT_OUT", pd.Series([0])) == 0]

    star_boost = safe_mean(starOut["MIN"]) - safe_mean(starIn["MIN"])
    res.append(star_missing * star_boost)

    # ============================
    # 20 — MIN_EXPECTATION_LOCATION
    # ============================
    home_df = player_df[player_df["HOME_GAME"] == 1]
    away_df = player_df[player_df["HOME_GAME"] == 0]

    home_min = safe_mean(home_df["MIN"])
    away_min = safe_mean(away_df["MIN"])

    res.append(home_flag * (home_min - avg_min) + (1 - home_flag) * (away_min - avg_min))

    # ============================
    # 21 — MATCHUP_MIN_DELTA
    # ============================
    res.append(safe_delta(matchup_df["MIN"], avg_min))

    # ============================
    # 22 — TEAM_PACE_OVER_LEAGUE_AVG
    # ============================
    league_pace_avg = safe_mean(league_df["PACE"]) if "PACE" in league_df.columns else 100.0
    team_pace_over_league = safe_mean(team_df["TEAM_PACE"]) - league_pace_avg
    res.append(team_pace_over_league)

    # ============================
    # 23 — EXPECTED_PACE
    # ============================
    expected_pace = (safe_mean(team_df["TEAM_PACE"]) + safe_mean(opp_team_df["TEAM_PACE"])) / 2
    res.append(expected_pace)

    # ============================
    # 24 — EXPECTED_PACE_X_MIN (new)
    # ============================
    expected_pace_x_min = round(expected_pace * avg_min, 2)
    res.append(expected_pace_x_min)

    # ============================
    # 25 — OPP_DEF_RATING_OVER_LEAGUE_AVG
    # ============================
    league_def_avg = safe_mean(league_df["DEF_RATING"]) if "DEF_RATING" in league_df.columns else 110.0
    opp_def_avg = safe_mean(opp_team_df["TEAM_DEF_RATING"])
    opp_def_rating_over_league = opp_def_avg - league_def_avg
    res.append(opp_def_rating_over_league)

    # ============================
    # 26 — OPP_DEF_RATING_OVER_LEAGUE_x_MIN (new)
    # ============================
    opp_def_rating_over_league_x_min = round(opp_def_rating_over_league * avg_min, 2)
    res.append(opp_def_rating_over_league_x_min)

    # ============================
    # 27 — OPP_PACE_OVER_LEAGUE_AVG
    # ============================
    opp_pace_over_league = safe_mean(opp_team_df["TEAM_PACE"]) - league_pace_avg
    res.append(opp_pace_over_league)

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
