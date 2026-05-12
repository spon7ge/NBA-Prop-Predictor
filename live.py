import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from src.pipeline.props_pipeline.min_pipeline import min_pipeline
from src.pipeline.props_pipeline.ppm_pipeline import ppm_pipeline
from src.pipeline.props_pipeline.apm_pipeline import apm_pipeline
from src.pipeline.props_pipeline.rpm_pipeline import rpm_pipeline
from src.utils.team_info import *
from src.utils.scrap_starters import NBADailyLineups
from src.utils.distributions import (
    run_pts_simulation,
    run_count_simulation_nbinom as run_count_simulation,
)
from src.utils.context import (
    adjust_predictions,
    get_game_context,
)
from src.utils.slates import (
    line_probs_for_market,
    build_dfs_slates_from_aligned,
)
from src.utils.helpers import *


# Populated at startup by main(); empty default prevents import-time side-effects
outPlayers: dict = {}


# ---------------------------------------------------------
# DAILY PIPELINE
# ---------------------------------------------------------

def main(
    *,
    min_ev: float = 0.5,
    min_kelly: float = 0.10,
    kelly_fraction: float = 0.5,
    top_n: int = 10,
    n_sims: int = 10_000,
    bookmakers: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Run the full daily prop-prediction pipeline end-to-end:
      1. Scrape lineups & injury updates
      2. Load game logs, team odds, player lines, and models
      3. Predict MIN × RATE quantiles for PTS / AST / REB
      4. Adjust predictions with player_scenarios + game_context
      5. Simulate line probabilities
      6/7. Enrich DFS picks via ``enrich_dfs_picks`` (sharp context + model probs) → all_line_probs
      8. Build greedy 2-/3-leg slates per bookmaker from dfs_sharp_aligned picks
      9. Append predictions & slates to the ledger (`log.snapshot`)
    """
    from src.utils.helpers import (
        load_base_df, load_team_odds, load_player_lines,
        load_models, merge_with_bookmaker,
        BOOKMAKER_SLATE_PATHS, BOOKMAKER_3LEG_PATHS,
    )

    if bookmakers is None:
        bookmakers = list(BOOKMAKER_SLATE_PATHS.keys())

    today_str    = datetime.today().strftime('%Y%m%d')
    current_date = datetime.today().strftime('%Y-%m-%d')

    # ── 1. Lineups & injury scrape ───────────────────────────────────────────
    global outPlayers
    print("\n── Scraping lineups ──")
    scraper = NBADailyLineups("https://www.rotowire.com/basketball/nba-lineups.php")
    scraper.getDict()
    outPlayers = scraper.getOutPlayers()
    scraper.updateTeamInfo()
    if verbose:
        print(f"  Out players: {outPlayers}")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    print("\n── Loading data ──")
    base_df   = load_base_df()
    team_odds = load_team_odds()
    lines_dfs, lines_us = load_player_lines(today_str)

    print(f"Lines for date: {current_date}")

    lines_dfs_pts = lines_dfs[lines_dfs['CATEGORY'] == 'player_points']
    lines_dfs_ast = lines_dfs[lines_dfs['CATEGORY'] == 'player_assists']
    lines_dfs_reb = lines_dfs[lines_dfs['CATEGORY'] == 'player_rebounds']
    lines_us_pts  = lines_us[lines_us['CATEGORY']  == 'player_points']
    lines_us_ast  = lines_us[lines_us['CATEGORY']  == 'player_assists']
    lines_us_reb  = lines_us[lines_us['CATEGORY']  == 'player_rebounds']

    pts_names = lines_dfs_pts['NAME'].unique()
    ast_names = lines_dfs_ast['NAME'].unique()
    reb_names = lines_dfs_reb['NAME'].unique()

    # ── 3. Load models ───────────────────────────────────────────────────────
    print("\n── Loading models ──")
    models = load_models()
    min_q_models = models["min"]["quantile_models"]
    ppm_q_models = models["ppm"]["quantile_models"]
    apm_q_models = models["apm"]["quantile_models"]
    rpm_q_models = models["rpm"]["quantile_models"]

    # ── 4. Predict ───────────────────────────────────────────────────────────
    print("\n── Running predictions ──")
    pts_preds = predict_min_times_rate(
        pts_names, base_df, base_df, current_date,
        rate_pipeline=ppm_pipeline,
        rate_quantile_models=ppm_q_models,
        min_quantile_models=min_q_models,
        stat_prefix="PTS",
        verbose=verbose,
    )
    ast_preds = predict_min_times_rate(
        ast_names, base_df, base_df, current_date,
        rate_pipeline=apm_pipeline,
        rate_quantile_models=apm_q_models,
        min_quantile_models=min_q_models,
        stat_prefix="AST",
        verbose=verbose,
    )
    reb_preds = predict_min_times_rate(
        reb_names, base_df, base_df, current_date,
        rate_pipeline=rpm_pipeline,
        rate_quantile_models=rpm_q_models,
        min_quantile_models=min_q_models,
        stat_prefix="REB",
        verbose=verbose,
    )

    # ── 5. Context adjustments ───────────────────────────────────────────────
    print("\n── Adjusting predictions ──")
    def _names(df): return set(df["PLAYER_NAME"].tolist()) if "PLAYER_NAME" in df.columns else set()
    all_names = _names(pts_preds) | _names(ast_preds) | _names(reb_preds)
    game_contexts = {
        name: get_game_context(base_df, name, team_odds, is_playoff=True, out_players=outPlayers)
        for name in all_names
    }
    pts_preds = adjust_predictions(pts_preds, base_df, game_contexts, verbose=verbose)
    ast_preds = adjust_predictions(ast_preds, base_df, game_contexts, verbose=verbose)
    reb_preds = adjust_predictions(reb_preds, base_df, game_contexts, verbose=verbose)

    # ── 6. Simulate line probabilities ───────────────────────────────────────
    print("\n── Simulating line probabilities ──")
    raw_probs = pd.concat([
        line_probs_for_market(pts_preds, lines_dfs_pts, run_pts_simulation, n_sims=n_sims),
        line_probs_for_market(ast_preds, lines_dfs_ast, run_count_simulation, n_sims=n_sims),
        line_probs_for_market(reb_preds, lines_dfs_reb, run_count_simulation, n_sims=n_sims),
    ], ignore_index=True)
    print(f"  {len(raw_probs)} raw prop lines")

    # ── 6/7. Enrich DFS picks with sharp context & model probs ──────────────
    print("\n── Enriching DFS picks ──")
    from src.utils.generalized_best_bets_v2 import enrich_dfs_picks as _enrich_dfs_picks

    _circa_files = list(Path("data/props/circa+betonline_team_lines").glob("circa+betonline_*.json"))
    _team_odds_src = str(max(_circa_files, key=lambda f: f.stat().st_mtime)) if _circa_files else None

    enriched_path, aligned_path, all_line_probs = _enrich_dfs_picks(
        dfs_df=lines_dfs,
        us_df=lines_us,
        base_df=base_df,
        all_line_probs=raw_probs,
        team_odds_source=_team_odds_src,
        dfs_platforms=None,
        current_date=current_date,
        verbose=verbose,
    )

    if all_line_probs.empty:
        print("\nNo enriched lines — skipping snapshot")
        return

    # ── 8. Build slates from dfs_sharp_aligned picks ─────────────────────────
    print("\n── Building slates ──")
    for bk in bookmakers:
        build_dfs_slates_from_aligned(
            aligned_path,
            platform=bk,
            out_2leg=BOOKMAKER_SLATE_PATHS.get(bk),
            out_3leg=BOOKMAKER_3LEG_PATHS.get(bk),
            stake_dollars=10.0,
            top_n=top_n,
            kelly_fraction=kelly_fraction,
            verbose=verbose,
        )

    # ── 9. Log to ledger ─────────────────────────────────────────────────────
    print("\n── Logging predictions & slates ──")
    from src.utils import log as _log

    # Rename key columns so log.snapshot can find its dedup/side fields,
    # then expand the `model` nested dict to flat columns alongside all
    # other enriched context (sharp, consensus, game_context, etc. stay as-is).
    _log_df = all_line_probs.rename(columns={
        "player":   "PLAYER_NAME",
        "market":   "MARKET",
        "dfs_line": "LINE",
        "platform": "LINE_BOOKMAKER",
    }).copy()
    if "model" in _log_df.columns:
        _model_flat = _log_df.pop("model").apply(pd.Series).rename(columns={
            "p_over":   "P_OVER",
            "p_under":  "P_UNDER",
            "min_q10":  "MIN_Q10",
            "min_q50":  "MIN_Q50",
            "min_q90":  "MIN_Q90",
            "stat_q10": "STAT_Q10",
            "stat_q50": "STAT_Q50",
            "stat_q90": "STAT_Q90",
        })
        _log_df = pd.concat([_log_df.reset_index(drop=True), _model_flat.reset_index(drop=True)], axis=1)

    slate_paths = {
        bk: {
            "2leg": BOOKMAKER_SLATE_PATHS.get(bk, ""),
            "3leg": BOOKMAKER_3LEG_PATHS.get(bk, ""),
        }
        for bk in bookmakers
    }
    _log.snapshot(_log_df, slate_paths, date=current_date)

    print("\n── Done ──")


if __name__ == "__main__":
    main()