import numpy as np
import pandas as pd

def min_pipeline(df, name):
    df = df.sort_values(["GAME_DATE", "PLAYER_ID"]).copy()

    if "MIN_roll5" in df.columns:
        df["TEAM_MIN_RANK_L5"] = (
            df.groupby(["TEAM_ID", "GAME_DATE"])["MIN_roll5"]
            .rank(ascending=False, method="dense")
        )
    if "MIN_roll10" in df.columns:
        df["TEAM_MIN_RANK_L10"] = (
            df.groupby(["TEAM_ID", "GAME_DATE"])["MIN_roll10"]
            .rank(ascending=False, method="dense")
        )
    if "PTS_roll10" in df.columns:
        df["TEAM_PTS_RANK_L10"] = (
            df.groupby(["TEAM_ID", "GAME_DATE"])["PTS_roll10"]
            .rank(ascending=False, method="dense")
        )
    if "USG_PCT_roll10" in df.columns:
        df["TEAM_USG_RANK_L10"] = (
            df.groupby(["TEAM_ID", "GAME_DATE"])["USG_PCT_roll10"]
            .rank(ascending=False, method="dense")
        )

    df["MIN_ewm3"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
    )
    df["MIN_ewm5"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )
    df["MIN_ewm10"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).ewm(span=10, adjust=False).mean()
    )

    df["MIN_season_avg"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"])["MIN"].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df["MEDIAN_MIN_L20"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).rolling(20).median()
    )

    team_game = (
        df[["TEAM_ID", "GAME_ID", "GAME_DATE", "TEAM_POSS"]]
        .drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
        .sort_values(["TEAM_ID", "GAME_DATE"])
    )
    team_game["TEAM_POSS_roll5"] = (
        team_game.groupby("TEAM_ID")["TEAM_POSS"]
        .rolling(5, min_periods=1).mean().round(2)
        .reset_index(level=0, drop=True)
    )
    df = df.drop(columns=["TEAM_POSS_roll5"], errors="ignore")
    df = df.join(
        team_game.set_index(["TEAM_ID", "GAME_ID"])[["TEAM_POSS_roll5"]],
        on=["TEAM_ID", "GAME_ID"],
    )

    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE")
    if len(pdf) < 10:
        return None

    last = pdf.iloc[-1]

    def g(col, default=np.nan):
        if col not in pdf.columns:
            return default
        v = last[col]
        return float(v) if pd.notna(v) else default

    starter_x_days = float(last["STARTING"]) * float(g("DAYS_REST", 0))

    res = []

    res.append(int(last["STARTING"]))
    res.append(round(float(last["MIN_ewm5"]), 2))
    res.append(float(g("TEAM_MIN_RANK_L5")))
    res.append(round(float(last["MIN_ewm10"]), 2))
    res.append(float(g("TEAM_MIN_RANK_L10")))
    res.append(round(float(last["MIN_ewm3"]), 2))
    res.append(round(float(g("MIN_season_avg")), 2))
    res.append(float(g("TEAM_PTS_RANK_L10")))
    res.append(float(g("TEAM_POSS_roll5")))
    res.append(int(pdf["STARTING"].iloc[-2]))
    res.append(float(pdf["STARTING"].tail(10).mean()))
    res.append(int(g("STAR_SAT_OUT", 0)))
    res.append(int(pdf["STARTING"].iloc[-3]))
    res.append(round(float(pdf["MIN"].iloc[-2]), 2))
    res.append(float(g("TEAM_USG_RANK_L10")))
    res.append(starter_x_days)
    res.append(float(g("MEDIAN_MIN_L20")))

    return res