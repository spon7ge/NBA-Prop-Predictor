import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp
from src.utils.team_info import projectedStartingFive


def apm_pipeline(df, name, date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]
    player_team = last["TEAM_ABBREVIATION"]

    def _ewm_last(col: str, span: int) -> float:
        return float(pdf[col].astype(float).ewm(span=span, adjust=False).mean().iloc[-1])

    # ── Opponent setup ────────────────────────────────────────────────────────
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])

    # ── Team setup ────────────────────────────────────────────────────────────
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]

    # ── AST_PER_MIN_season_avg ────────────────────────────────────────────────
    ast_per_min_season_avg = float(pdf["AST_PER_MIN"].mean())
    res.append(ast_per_min_season_avg if pd.notna(ast_per_min_season_avg) else float("nan"))

    # ── AST_10_ewm ────────────────────────────────────────────────────────────
    ast_10_ewm = _ewm_last("AST", 10)
    res.append(ast_10_ewm if pd.notna(ast_10_ewm) else float("nan"))

    # ── TCHS_PER_MIN_10_ewm ───────────────────────────────────────────────────
    tchs_per_min_10_ewm = (
        _ewm_last("TCHS_PER_MIN", 10) if "TCHS_PER_MIN" in pdf.columns else float("nan")
    )
    res.append(tchs_per_min_10_ewm if pd.notna(tchs_per_min_10_ewm) else float("nan"))

    # ── AST_PCT_10_ewm ────────────────────────────────────────────────────────
    ast_pct_10_ewm = _ewm_last("AST_PCT", 10)
    res.append(ast_pct_10_ewm if pd.notna(ast_pct_10_ewm) else float("nan"))

    # ── PASS_PER_MIN_5_ewm ────────────────────────────────────────────────────
    pass_per_min_5_ewm = (
        _ewm_last("PASS_PER_MIN", 5) if "PASS_PER_MIN" in pdf.columns else float("nan")
    )
    res.append(pass_per_min_5_ewm if pd.notna(pass_per_min_5_ewm) else float("nan"))

    # ── OPP_TEAM_AST_ALLOWED ──────────────────────────────────────────────────
    opp_team_ast_allowed = (
        float(opp_team["OPP_AST"].astype(float).mean())
        if "OPP_AST" in opp_team.columns else float("nan")
    )
    res.append(opp_team_ast_allowed if pd.notna(opp_team_ast_allowed) else float("nan"))

    # ── POSITION_ENCODED ──────────────────────────────────────────────────────
    res.append(float(pdf["POSITION_ENCODED"].iloc[-1]) if "POSITION_ENCODED" in pdf.columns else float("nan"))

    # ── TEAM_AST_PER_MIN_RANK_L10 ─────────────────────────────────────────────
    team_ast_rank = float(pdf["TEAM_AST_PER_MIN_RANK_L10"].iloc[-1]) if "TEAM_AST_PER_MIN_RANK_L10" in pdf.columns else float("nan")
    res.append(team_ast_rank if pd.notna(team_ast_rank) else float("nan"))

    # ── FTAST_PER_MIN_season_avg ──────────────────────────────────────────────
    ftast_per_min_season_avg = (
        float(pdf["FTAST_PER_MIN"].mean()) if "FTAST_PER_MIN" in pdf.columns else float("nan")
    )
    res.append(ftast_per_min_season_avg if pd.notna(ftast_per_min_season_avg) else float("nan"))

    # ── AST_PER_MIN_lag1 ──────────────────────────────────────────────────────
    ast_per_min_lag1 = float(pdf["AST_PER_MIN"].iloc[-1])
    res.append(ast_per_min_lag1 if pd.notna(ast_per_min_lag1) else float("nan"))

    # ── FTAST_PER_MIN_10_ewm ──────────────────────────────────────────────────
    ftast_per_min_10_ewm = (
        _ewm_last("FTAST_PER_MIN", 10) if "FTAST_PER_MIN" in pdf.columns else float("nan")
    )
    res.append(ftast_per_min_10_ewm if pd.notna(ftast_per_min_10_ewm) else float("nan"))

    # ── IS_PLAYOFF ────────────────────────────────────────────────────────────
    is_playoff = float(last["IS_PLAYOFF"]) if "IS_PLAYOFF" in last.index else float("nan")
    res.append(is_playoff)

    return res
