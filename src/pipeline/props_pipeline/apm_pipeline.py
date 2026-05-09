import pandas as pd

from src.utils.helper_functions import findOpp


def _shifted_ewm_tail(col: pd.Series, span: int, *, decimals: int = 2) -> float:
    v = (
        col.astype(float)
        .shift(1)
        .ewm(span=span, adjust=False)
        .mean()
        .iloc[-1]
    )
    if pd.isna(v):
        return float("nan")
    return round(float(v), decimals) if decimals is not None else float(v)


def _opp_team_ast_allowed(
    df: pd.DataFrame, opp_abbr: str, *, as_of_date: str | None
) -> float:
    """Match training `OPP_TEAM_AST_ALLOWED`: prior expanding mean of OPP_AST on opponent team rows."""
    if "OPP_AST" not in df.columns:
        return float("nan")
    opp = df[df["TEAM_ABBREVIATION"] == opp_abbr].drop_duplicates(
        subset=["TEAM_ID", "GAME_ID"]
    )
    if opp.empty:
        return float("nan")
    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date)
        dated = opp[pd.to_datetime(opp["GAME_DATE"], errors="coerce") < cutoff]
        opp = dated if len(dated) > 0 else opp
    opp = opp.sort_values(["GAME_DATE"]).copy()
    if "SEASON_YEAR" not in opp.columns:
        return float(
            opp["OPP_AST"]
            .astype(float)
            .shift(1)
            .expanding()
            .mean()
            .round(3)
            .iloc[-1]
        )
    last = opp.iloc[-1]
    mask = opp["SEASON_YEAR"] == last["SEASON_YEAR"]
    season = opp.loc[mask]
    allowed = (
        season.groupby(["TEAM_ID", "SEASON_YEAR"], sort=False)["OPP_AST"]
        .transform(lambda x: x.shift(1).expanding().mean().round(3))
    )
    return float(season.assign(_a=allowed)["_a"].iloc[-1])


def apm_pipeline(df, name, date):
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]

    # ── Opponent setup ──────────────────────────────────────────────────────────
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)

    # ── 1. AST_PER_MIN_season_avg (shift(1).expanding mean within season) ──────
    if "SEASON_YEAR" in pdf.columns:
        sx = pdf.groupby(["PLAYER_ID", "SEASON_YEAR"], group_keys=False)[
            "AST_PER_MIN"
        ].transform(lambda x: x.shift(1).expanding().mean().round(2))
        ast_per_min_season_avg = float(sx.iloc[-1]) if pd.notna(sx.iloc[-1]) else float("nan")
    else:
        ast_per_min_season_avg = float(
            pdf["AST_PER_MIN"].astype(float).shift(1).expanding().mean().round(2).iloc[-1]
        )
        if pd.isna(ast_per_min_season_avg):
            ast_per_min_season_avg = float("nan")
    res.append(ast_per_min_season_avg)

    # ── 2. TEAM_AST_PER_MIN_RANK_L10 ───────────────────────────────────────────
    # Same spirit as notebook: AST_PER_MIN_roll10 = shift(1).rolling(10).mean —
    # for predicting the next slate that equals mean(AST_PER_MIN) over last 10 games.
    team_df = df[df["TEAM_ID"] == last["TEAM_ID"]]
    teammate_roll = {}
    for player, grp in team_df.groupby("PLAYER_NAME"):
        grp_sorted = grp.sort_values("GAME_DATE")
        roll_avg = grp_sorted["AST_PER_MIN"].astype(float).tail(10).mean()
        teammate_roll[player] = roll_avg
    sorted_teammates = sorted(teammate_roll.items(), key=lambda x: x[1], reverse=True)
    rank_map = {p: float(i + 1) for i, (p, _) in enumerate(sorted_teammates)}
    team_ast_rank = rank_map.get(name, float("nan"))
    res.append(team_ast_rank if pd.notna(team_ast_rank) else float("nan"))

    # ── 3. OPP_TEAM_AST_ALLOWED ────────────────────────────────────────────────
    if "OPP_TEAM_AST_ALLOWED" in last.index and pd.notna(last.get("OPP_TEAM_AST_ALLOWED")):
        opp_team_ast_allowed = float(last["OPP_TEAM_AST_ALLOWED"])
    else:
        opp_team_ast_allowed = _opp_team_ast_allowed(df, opp_abbr, as_of_date=date)
    res.append(
        opp_team_ast_allowed if pd.notna(opp_team_ast_allowed) else float("nan")
    )

    # ── 4. PASS_PER_MIN_5_ewm ───────────────────────────────────────────────────
    pass_per_min_5_ewm = (
        _shifted_ewm_tail(pdf["PASS_PER_MIN"], 5)
        if "PASS_PER_MIN" in pdf.columns
        else float("nan")
    )
    res.append(pass_per_min_5_ewm)

    # ── 5. POSITION_ENCODED ─────────────────────────────────────────────────────
    res.append(
        float(pdf["POSITION_ENCODED"].iloc[-1])
        if "POSITION_ENCODED" in pdf.columns
        else float("nan")
    )

    # ── 6. AST_PER_MIN_STD_SEASON ───────────────────────────────────────────────
    ast_std_series = pdf.groupby("PLAYER_ID", group_keys=False)["AST_PER_MIN"].transform(
        lambda x: x.shift(1).expanding(min_periods=2).std()
    )
    ast_per_min_std_season = float(ast_std_series.iloc[-1]) if pd.notna(ast_std_series.iloc[-1]) else float("nan")
    res.append(ast_per_min_std_season)

    # ── 7. APM_TREND ────────────────────────────────────────────────────────────
    ast_per_min_5_ewm = _shifted_ewm_tail(pdf["AST_PER_MIN"], 5)
    apm_trend = (
        round(ast_per_min_5_ewm, 2) - ast_per_min_season_avg
        if pd.notna(ast_per_min_5_ewm) and pd.notna(ast_per_min_season_avg)
        else float("nan")
    )
    res.append(apm_trend)

    return res
