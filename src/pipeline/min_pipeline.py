import pandas as pd

from src.features.min_features import MIN_FEATURES


def _consec_starts(starting_series: pd.Series) -> float:
    """Consecutive-starts streak at the end of the series (current state, unshifted)."""
    vals = starting_series.astype(float).values
    count = 0
    for val in reversed(vals):
        if val == 1.0:
            count += 1
        else:
            break
    return float(count)


def _team_min_rank_l10(df: pd.DataFrame, player_name: str, team_id) -> float:
    """
    Rank of the player's L10 avg minutes among all teammates (unshifted).
    Rank 1 = most minutes on the team.
    """
    team_df = df[df["TEAM_ID"] == team_id].copy()
    if team_df.empty:
        return float("nan")

    l10_avg = (
        team_df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME")["MIN"]
        .apply(lambda s: s.astype(float).tail(10).mean())
    )
    if l10_avg.empty or player_name not in l10_avg.index:
        return float("nan")

    ranked = l10_avg.rank(ascending=False, method="min")
    return float(ranked[player_name])


def _team_usg_rank_l10(df: pd.DataFrame, player_name: str, team_id) -> float:
    """
    Rank of the player's L10 avg usage among all teammates (unshifted).
    Rank 1 = highest usage on the team.
    Expects a USG (or USG_PCT) column in df.
    """
    usg_col = "USG_PCT" if "USG_PCT" in df.columns else "USG"
    if usg_col not in df.columns:
        return float("nan")

    team_df = df[df["TEAM_ID"] == team_id].copy()
    if team_df.empty:
        return float("nan")

    l10_avg = (
        team_df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME")[usg_col]
        .apply(lambda s: s.astype(float).tail(10).mean())
    )
    if l10_avg.empty or player_name not in l10_avg.index:
        return float("nan")

    ranked = l10_avg.rank(ascending=False, method="min")
    return float(ranked[player_name])


def min_pipeline(df, name, current_date=None):
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    last = pdf.iloc[-1]
    min_s = pdf["MIN"].astype(float)

    # MIN_10_ewm — unshifted ewm(span=10), includes latest game
    min_10_ewm_val = float(min_s.ewm(span=10, adjust=False).mean().iloc[-1])

    # MIN_SEASON_MEAN
    season_mean = float(min_s.mean())

    # MIN_SEASON_STD
    season_std = float(min_s.std()) if len(min_s) > 1 else float("nan")

    # STARTER_ROLL10_PCT — unshifted rolling mean of last 10 games
    if "STARTING" not in pdf.columns:
        starter_roll10 = float("nan")
        consec = float("nan")
    else:
        st = pdf["STARTING"].astype(float)
        n = min(10, len(st))
        starter_roll10 = round(float(st.tail(n).mean()), 2)
        consec = _consec_starts(st)

    # MIN_RATE_OF_CHANGE — fractional deviation of recent form (MIN_10_ewm) from season baseline
    if pd.notna(season_mean) and season_mean != 0 and pd.notna(min_10_ewm_val):
        min_roc = float((min_10_ewm_val - season_mean) / season_mean)
    else:
        min_roc = float("nan")

    # TEAM_MIN_RANK_L10
    team_min_rank = _team_min_rank_l10(df, name, last["TEAM_ID"])

    # TEAM_USG_RANK_L10
    team_usg_rank = _team_usg_rank_l10(df, name, last["TEAM_ID"])

    # MIN_P10_L10, MIN_P90_L10, MIN_STD_L10 — percentiles/std of last 10 games
    min_l10 = min_s.tail(10)
    min_p10 = float(min_l10.quantile(0.10))
    min_p90 = float(min_l10.quantile(0.90))
    min_std_l10 = float(min_l10.std()) if len(min_l10) > 1 else float("nan")

    # SPD_10_ewm — unshifted ewm(span=10) of SPD column
    if "SPD" in pdf.columns:
        spd_s = pdf["SPD"].astype(float)
        spd_10_ewm_val = float(spd_s.ewm(span=10, adjust=False).mean().iloc[-1])
    else:
        spd_10_ewm_val = float("nan")

    return [
        min_10_ewm_val,
        season_mean,
        starter_roll10,
        consec,
        min_roc,
        team_min_rank,
        team_usg_rank,
        min_p10,
        min_p90,
        min_std_l10,
        season_std,
        spd_10_ewm_val,
    ]