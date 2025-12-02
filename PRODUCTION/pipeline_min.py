from datetime import datetime, timedelta
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, scheduleleaguev2, leaguedashteamstats
from PRODUCTION.teamInfo import mainStartingFive, teamStarPlayer, projectedStartingFive, nameDict
from PRODUCTION.helperFunctions import findOpp, getUpcomingGamesCached

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
    # 1 — Starting
    # ============================
    res.append(int(player_name in projectedStartingFive[team]))

    # ============================
    # 2 — Star Sat Out
    # ============================
    star_sat_out = int(teamStarPlayer[team] not in projectedStartingFive[team])
    res.append(star_sat_out)

    # ============================
    # 3 — Days Rest
    # ============================
    last_game = pd.to_datetime(player_df["GAME_DATE"]).max()
    days_rested = (pd.to_datetime(current_date) - last_game).days
    res.append(days_rested)

    # ============================
    # 4 — Usual Starters Available
    # ============================
    main_starters = set(mainStartingFive[team])
    projected_starters = set(projectedStartingFive[team])
    missing_starters = len(main_starters - projected_starters)
    res.append(missing_starters)

    # ============================
    # NEW FEATURES — MIN Ceiling/Floor L5 & Interactions
    # ============================
    min_last_5 = player_df["MIN"].tail(5)
    min_ceiling_l5 = float(min_last_5.max()) if min_last_5.size > 0 else 0.0
    min_floor_l5 = float(min_last_5.min()) if min_last_5.size > 0 else 0.0
    
    # Calculate avg_min first (needed for deltas)
    avg_min = safe_mean(player_df["MIN"])
    
    # Calculate deltas
    min_ceiling_l5_delta = round(min_ceiling_l5 - avg_min, 2) if min_last_5.size > 0 else 0.0
    min_floor_l5_delta = round(min_floor_l5 - avg_min, 2) if min_last_5.size > 0 else 0.0
    
    # Calculate interaction features
    starting_flag = int(player_name in projectedStartingFive[team])
    starting_x_min_avg = round(starting_flag * avg_min, 2)
    starting_x_min_ceiling = round(starting_flag * min_ceiling_l5, 2)
    
    res.append(min_ceiling_l5_delta)
    res.append(min_floor_l5_delta)
    res.append(starting_x_min_avg)
    res.append(starting_x_min_ceiling)

    # ============================
    # 7 — MIN averages & trend
    # ============================
    res.append(avg_min)
    res.append(safe_delta(player_df["MIN"].tail(5), avg_min))
    res.append(safe_delta(player_df["MIN"].tail(7), avg_min))
    res.append(safe_delta(player_df["MIN"].tail(10), avg_min))

    # ============================
    # 8 — Variability
    # ============================
    res.append(safe_std(player_df["MIN"].tail(5)))
    res.append(safe_std(player_df["MIN"].tail(10)))

    # ============================
    # 9 — Star Out Boost
    # ============================
    star_missing = int(teamStarPlayer[team] not in projectedStartingFive[team])
    starOut = player_df[player_df["STAR_SAT_OUT"] == 1]
    starIn = player_df[player_df["STAR_SAT_OUT"] == 0]

    star_boost = safe_mean(starOut["MIN"]) - safe_mean(starIn["MIN"])
    res.append(star_missing * star_boost)

    # ============================
    # 10 — SPD L5 and L10 Over Baseline
    # ============================
    if "SPD" in player_df.columns:
        spd_avg_to_date = safe_mean(player_df["SPD"])
        spd_l5 = safe_mean(player_df["SPD"].tail(5))
        spd_l10 = safe_mean(player_df["SPD"].tail(10))
        
        # Calculate ratios (over baseline)
        epsilon = 1e-8
        spd_l5_over_baseline = round(spd_l5 / (spd_avg_to_date + epsilon), 2) if spd_avg_to_date > 0 else 1.0
        spd_l10_over_baseline = round(spd_l10 / (spd_avg_to_date + epsilon), 2) if spd_avg_to_date > 0 else 1.0
    else:
        # If SPD column doesn't exist, use default values
        spd_l5_over_baseline = 1.0
        spd_l10_over_baseline = 1.0
    
    res.append(spd_l5_over_baseline)
    res.append(spd_l10_over_baseline)

    # ============================
    # 11 — Location: Home / Away
    # ============================
    home_df = player_df[player_df["HOME_GAME"] == 1]
    away_df = player_df[player_df["HOME_GAME"] == 0]

    home_min = safe_mean(home_df["MIN"])
    away_min = safe_mean(away_df["MIN"])

    res.append(home_flag * (home_min - avg_min) + (1 - home_flag) * (away_min - avg_min))

    # ============================
    # 12 — Matchup MIN Delta
    # ============================
    res.append(safe_delta(matchup_df["MIN"], avg_min))

    # ============================
    # 13 — Team Pace vs League Pace
    # ============================
    league_pace_avg = safe_mean(league_df["PACE"])
    res.append(safe_mean(team_df["TEAM_PACE"]) - league_pace_avg)

    # Expected Pace (team + opp) / 2
    expected_pace = (safe_mean(team_df["TEAM_PACE"]) + safe_mean(opp_team_df["TEAM_PACE"])) / 2
    res.append(expected_pace)

    # ============================
    # 14 — Opponent Defensive Context
    # ============================
    league_def_avg = safe_mean(league_df["DEF_RATING"])
    opp_def_avg = safe_mean(opp_team_df["TEAM_DEF_RATING"])

    res.append(opp_def_avg - league_def_avg)
    res.append(safe_mean(opp_team_df["TEAM_PACE"]) - league_pace_avg)

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
