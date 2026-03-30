from src.utils.team_info import *
from src.utils.helper_functions import *

def ppm_pipeline(df, name, date):
    df = df.sort_values(["GAME_DATE", "PLAYER_ID"]).copy()
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE")
    pid = int(df.loc[df["PLAYER_NAME"] == name, "PLAYER_ID"].iloc[-1])
    res = []
    pts_ewm_l10 = df.groupby('PLAYER_ID')['PTS'].apply(lambda x: x.tail(10).ewm(span=10, adjust=False).mean().iloc[-1])
    pts_ewm_l5 = df.groupby('PLAYER_ID')['PTS'].apply(lambda x: x.tail(5).ewm(span=5, adjust=False).mean().iloc[-1])
    pts_ewm_l3 = df.groupby('PLAYER_ID')['PTS'].apply(lambda x: x.tail(3).ewm(span=3, adjust=False).mean().iloc[-1])
    res.append(pts_ewm_l10[pid])
    res.append(pts_ewm_l5[pid])
    res.append(pts_ewm_l3[pid])

    pts_per_min_ewm_l10 = df.groupby('PLAYER_ID')['PTS_PER_MIN'].apply(lambda x: x.tail(10).ewm(span=10, adjust=False).mean().iloc[-1])
    pts_per_min_ewm_l5 = df.groupby('PLAYER_ID')['PTS_PER_MIN'].apply(lambda x: x.tail(5).ewm(span=5, adjust=False).mean().iloc[-1])
    pts_per_min_ewm_l3 = df.groupby('PLAYER_ID')['PTS_PER_MIN'].apply(lambda x: x.tail(3).ewm(span=3, adjust=False).mean().iloc[-1])
    res.append(pts_per_min_ewm_l10[pid])
    res.append(pts_per_min_ewm_l5[pid])
    res.append(pts_per_min_ewm_l3[pid])

    fga_per_min_ewm_l10 = df.groupby('PLAYER_ID')['FGA_PER_MIN'].apply(lambda x: x.tail(10).ewm(span=10, adjust=False).mean().iloc[-1])
    fga_per_min_ewm_l5 = df.groupby('PLAYER_ID')['FGA_PER_MIN'].apply(lambda x: x.tail(5).ewm(span=5, adjust=False).mean().iloc[-1])
    fga_per_min_ewm_l3 = df.groupby('PLAYER_ID')['FGA_PER_MIN'].apply(lambda x: x.tail(3).ewm(span=3, adjust=False).mean().iloc[-1])
    res.append(fga_per_min_ewm_l10[pid])
    res.append(fga_per_min_ewm_l5[pid])
    res.append(fga_per_min_ewm_l3[pid])

    fg3a_per_min_ewm_l10 = df.groupby('PLAYER_ID')['3PA_PER_MIN'].apply(lambda x: x.tail(10).ewm(span=10, adjust=False).mean().iloc[-1])
    fg3a_per_min_ewm_l5 = df.groupby('PLAYER_ID')['3PA_PER_MIN'].apply(lambda x: x.tail(5).ewm(span=5, adjust=False).mean().iloc[-1])
    fg3a_per_min_ewm_l3 = df.groupby('PLAYER_ID')['3PA_PER_MIN'].apply(lambda x: x.tail(3).ewm(span=3, adjust=False).mean().iloc[-1])
    res.append(fg3a_per_min_ewm_l10[pid])
    res.append(fg3a_per_min_ewm_l5[pid])
    res.append(fg3a_per_min_ewm_l3[pid])

    usg_pct_l10 = df.groupby('PLAYER_ID')['USG_PCT'].apply(lambda x: x.tail(10).mean())
    usg_pct_l5 = df.groupby('PLAYER_ID')['USG_PCT'].apply(lambda x: x.tail(5).mean())
    usg_pct_l3 = df.groupby('PLAYER_ID')['USG_PCT'].apply(lambda x: x.tail(3).mean())
    res.append(usg_pct_l10[pid])
    res.append(usg_pct_l5[pid])
    res.append(usg_pct_l3[pid])

    ts_pct_l10 = df.groupby('PLAYER_ID')['TS_PCT'].apply(lambda x: x.tail(10).mean())
    ts_pct_l3 = df.groupby('PLAYER_ID')['TS_PCT'].apply(lambda x: x.tail(3).mean())
    res.append(ts_pct_l10[pid])
    res.append(ts_pct_l3[pid])

    poss_roll_l10 = df.groupby('PLAYER_ID')['POSS_roll10'].apply(lambda x: x.tail(10).mean())
    res.append(poss_roll_l10[pid])

    # Opponent stats
    opp, _ = findOpp(name, pdf, date, max_days_ahead=3)
    opp_team = df[df['TEAM_ABBREVIATION'] == opp].sort_values("GAME_DATE")
    opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], inplace=True)
    opp_team_poss_roll5 = (
        opp_team.groupby("TEAM_ID")["TEAM_POSS"]
        .rolling(5, min_periods=1).mean().round(2)
        .reset_index(level=0, drop=True))
    opp_team_pace_roll5 = (
        opp_team.groupby("TEAM_ID")["TEAM_PACE"]
        .rolling(5, min_periods=1).mean().round(2)
        .reset_index(level=0, drop=True))
    opp_team_def_rating_roll5 = (
        opp_team.groupby("TEAM_ID")["TEAM_DEF_RATING"]
        .rolling(5, min_periods=1).mean().round(2)
        .reset_index(level=0, drop=True))
    res.append(float(opp_team_poss_roll5.iloc[-1]))
    res.append(float(opp_team_pace_roll5.iloc[-1]))
    res.append(float(opp_team_def_rating_roll5.iloc[-1]))

    # Proxies + team ranks (PPM_FEATURES order)
    last = df[df["PLAYER_ID"] == pid].sort_values("GAME_DATE").iloc[-1]
    res.append(float(last["TEAM_POSS_roll5"]))
    res.append(float(last["PTS_share_proxy"]))
    res.append(float(last["MIN_share_proxy"]))
    res.append(float(last["TEAM_PTS_RANK_L10"]))
    res.append(float(last["TEAM_PTS_RANK_L5"]))
    res.append(float(last["TEAM_USG_RANK_L10"]))
    res.append(float(last["TEAM_TS_RANK_L10"]))

    return res
