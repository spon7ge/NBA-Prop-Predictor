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


# ==========================================================
#                USAGE FEATURE BUILDER
# ==========================================================
def player_usg_features(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer, league_df):

    # ----------------------------
    # Player-level data
    # ----------------------------
    player_df = data[data["PLAYER_NAME"] == player_name].sort_values("GAME_DATE")
    if player_df.empty:
        print(f"[USG] No data found for {player_name}")
        return None

    team = player_df["TEAM_ABBREVIATION"].iloc[-1]

    home_df = player_df[player_df["HOME_GAME"] == 1]
    away_df = player_df[player_df["HOME_GAME"] == 0]

    # ----------------------------
    # Opponent info for the upcoming game
    # ----------------------------
    opp, home_flag = findOpp(player_name, data, current_date)
    if opp is None:
        print(f"[USG] No opponent found for {player_name}")
        return None

    opp_df = data[data["TEAM_ABBREVIATION"] == opp]
    if opp_df.empty:
        print(f"[USG] Opponent data missing for {opp}")
        return None

    matchup_df = player_df[player_df["OPP_ABBREVIATION"] == opp]

    # ----------------------------
    # Team datasets
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

    # ==========================================================
    # 1 — STARTING STATUS
    # ==========================================================
    res.append(int(player_name in projectedStartingFive[team]))

    # ==========================================================
    # 2 — PLAYER_IS_TEAM_STAR
    # ==========================================================
    player_is_team_star = int(player_name == teamStarPlayer.get(team, None))
    res.append(player_is_team_star)

    # ==========================================================
    # 3 — STAR_SAT_OUT
    # ==========================================================
    star_sat_out = int(teamStarPlayer.get(team, None) not in projectedStartingFive.get(team, []))
    res.append(star_sat_out)

    # ==========================================================
    # 4 — LINEUP_FGA_SHARE_AVG
    # ==========================================================
    # Calculate average FGA share of projected starters
    projected_starters = projectedStartingFive[team]
    lineup_fga_shares = []
    
    # Get team's average FGA to date
    team_fga_avg = safe_mean(team_df["TEAM_FGA"]) if "TEAM_FGA" in team_df.columns else 0.0
    
    for starter_name in projected_starters:
        starter_df = data[data["PLAYER_NAME"] == starter_name]
        if not starter_df.empty:
            starter_fga_avg = safe_mean(starter_df["FGA"])
            if team_fga_avg > 0:
                fga_share = (starter_fga_avg / team_fga_avg) * 100
                lineup_fga_shares.append(fga_share)
    
    lineup_fga_share_avg = round(safe_mean(pd.Series(lineup_fga_shares)), 2) if lineup_fga_shares else 0.0
    res.append(lineup_fga_share_avg)

    # ==========================================================
    # 5 — USG_PCT_AVG_TO_DATE
    # ==========================================================
    usg_avg = safe_mean(player_df["USG_PCT"])
    res.append(usg_avg)

    # ==========================================================
    # 6 — USG_PCT_L5_OVER_BASELINE
    # ==========================================================
    usg_l5 = safe_mean(player_df["USG_PCT"].tail(5))
    epsilon = 1e-8
    usg_l5_over_baseline = round(usg_l5 / (usg_avg + epsilon), 2) if usg_avg > 0 else 1.0
    res.append(usg_l5_over_baseline)

    # ==========================================================
    # 7 — USG_PER_MIN
    # ==========================================================
    min_avg = safe_mean(player_df["MIN"])
    usg_per_min = round((usg_avg / min_avg) + 0.001, 2) if min_avg > 0 else 0.0
    res.append(usg_per_min)

    return res


# ==========================================================
#   WRAPPER (used directly by your model inference)
# ==========================================================
def buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer, league_df):

    features = player_usg_features(
        player_name, data, current_date,
        projectedStartingFive, mainStartingFive,
        teamStarPlayer, league_df
    )

    if features is None:
        print(f"[USG] Failed to generate features for {player_name}")
        return None

    return [features]
