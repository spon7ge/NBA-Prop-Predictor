"""
Data-loading and bookmaker-merge helpers for the daily pipeline.
All functions are pure (no side-effects on import).
"""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.features.feature_engineer.apm_features import _detect_star_players
from src.utils.dataScraper import generalized_best_bets


# Mapping from odds-API category slug → model MARKET label
PROP_LABEL_MAP = {
    'player_points':                'PTS',
    'player_rebounds':              'REB',
    'player_assists':               'AST',
    'player_turnovers':             'TOV',
    'player_frees_attempts':        'FTA',
    'player_threes':                '3PM',
    'player_blocks':                'BLK',
    'player_steals':                'STL',
    'player_blocks_steals':         'BLK+STL',
    'player_points_rebounds_assists':'PTS+REB+AST',
    'player_points_rebounds':       'PTS+REB',
    'player_points_assists':        'PTS+AST',
    'player_rebounds_assists':      'REB+AST',
}

# Category slug → game-log columns (same semantics as dataScraper.cat_to_stat_cols)
_OVER_RATE_CAT_TO_STAT = {
    'player_points': ['PTS'],
    'player_rebounds': ['REB'],
    'player_assists': ['AST'],
    'player_turnovers': ['TOV'],
    'player_frees_attempts': ['FTA'],
    'player_threes': ['FG3M'],
    'player_blocks': ['BLK'],
    'player_steals': ['STL'],
    'player_blocks_steals': ['BLK', 'STL'],
    'player_points_rebounds_assists': ['PTS', 'REB', 'AST'],
    'player_points_rebounds': ['PTS', 'REB'],
    'player_points_assists': ['PTS', 'AST'],
    'player_rebounds_assists': ['REB', 'AST'],
}

# Bookmaker name → output JSON file path
BOOKMAKER_SLATE_PATHS = {
    'PrizePicks':       'data/props/ev_analysis/prizepicks.json',
    'Underdog':         'data/props/ev_analysis/underdog.json',
    'Betr DFS':         'data/props/ev_analysis/betr.json',
    'DraftKings Pick6': 'data/props/ev_analysis/draftKings.json',
}

BOOKMAKER_3LEG_PATHS = {
    'PrizePicks':       'data/props/ev_analysis/prizepicks_3leg.json',
    'Underdog':         'data/props/ev_analysis/underdog_3leg.json',
    'Betr DFS':         'data/props/ev_analysis/betr_3leg.json',
    'DraftKings Pick6': 'data/props/ev_analysis/draftKings_3leg.json',
}


def load_base_df() -> pd.DataFrame:
    """Load and combine season + playoff game logs for S25 and S26."""
    s25 = pd.read_csv('data/raw/season_stats/S25.csv').sort_values('GAME_DATE')
    p25 = pd.read_csv('data/raw/playoff_stats/P25.csv').sort_values('GAME_DATE')
    s25 = _detect_star_players(pd.concat([s25, p25], ignore_index=True))

    s26 = pd.read_csv('data/raw/season_stats/S26.csv').sort_values('GAME_DATE')
    p26 = pd.read_csv('data/raw/playoff_stats/P26.csv').sort_values('GAME_DATE')
    s26 = _detect_star_players(pd.concat([s26, p26], ignore_index=True))

    base_df = pd.concat([s25, s26], ignore_index=True)
    print(f"base_df: {len(base_df)} rows, {base_df['PLAYER_NAME'].nunique()} players")
    return base_df


def load_team_odds() -> pd.DataFrame:
    """Load the latest NBA team-odds JSON from data/raw/team_lines/."""
    files = list(Path('data/raw/team_lines').glob('NBA_*.json'))
    if not files:
        raise FileNotFoundError("No NBA_*.json found in data/raw/team_lines/")
    file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Team odds: {file.name}")
    try:
        return pd.read_json(file)
    except ValueError:
        with open(file) as f:
            data = json.load(f)
        return pd.json_normalize(data)


def load_player_lines(today_str: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the latest DFS and US player-prop CSVs for today.
    Returns (lines_dfs, lines_us).
    """
    if today_str is None:
        today_str = datetime.today().strftime('%Y%m%d')

    def _latest(folder: str, pattern: str) -> Path:
        files = list(Path(folder).glob(pattern))
        if not files:
            raise FileNotFoundError(f"No file matching '{pattern}' in {folder}")
        return max(files, key=lambda f: f.stat().st_mtime)

    dfs_file = _latest('data/raw/player_lines', f'NBA_DFS_{today_str}*.csv')
    us_file  = _latest('data/raw/player_lines', f'NBA_US_{today_str}*.csv')
    print(f"DFS lines: {dfs_file.name}  |  US lines: {us_file.name}")
    return pd.read_csv(dfs_file), pd.read_csv(us_file)


def load_models(models_dir: str = "src/models/saved_models") -> dict:
    """
    Load all quantile model bundles.
    Uses alphabetical sort on filename so the latest date wins when multiple
    versions exist (e.g. ppm_quantile_xgb_2026-01-01 < ppm_quantile_xgb_2026-03-01).
    Returns a dict with keys: 'min', 'ppm', 'apm', 'rpm'.
    """
    models_path = Path(models_dir)

    def _latest_bundle(prefix: str):
        files = sorted(models_path.glob(f"{prefix}_quantile_xgb_*.joblib"))
        if not files:
            raise FileNotFoundError(f"No model for prefix '{prefix}' in {models_dir}")
        chosen = files[-1]
        print(f"  {chosen.name}")
        return joblib.load(chosen)

    print("Loading models...")
    return {
        "min": _latest_bundle("min"),
        "ppm": _latest_bundle("ppm"),
        "apm": _latest_bundle("apm"),
        "rpm": _latest_bundle("rpm"),
    }


def merge_with_bookmaker(
    all_line_probs: pd.DataFrame,
    lines_dfs: pd.DataFrame,
    lines_us: pd.DataFrame,
    base_df: pd.DataFrame,
    team_odds: pd.DataFrame,
    bookmaker: str,
    *,
    debug_nans: bool = False,
) -> pd.DataFrame:
    """
    Run generalized_best_bets for one bookmaker and left-join the enriched
    contextual stats onto the model probabilities in all_line_probs.

    Returns an enriched DataFrame ready for build_greedy_slate(), or an empty
    DataFrame if no lines were found for the bookmaker today.

    Pass ``debug_nans=True`` to print why NaNs appear inside ``generalized_best_bets``.
    """
    _, _, final = generalized_best_bets(
        lines_dfs, base_df, lines_us, team_odds,
        line_bookmaker=bookmaker,
        debug_nans=debug_nans,
    )

    if final.empty or 'CATEGORY' not in final.columns:
        print(f"[{bookmaker}] No lines found — skipping")
        return pd.DataFrame()

    df = final.copy()
    df['CATEGORY'] = df['CATEGORY'].map(PROP_LABEL_MAP).fillna(df['CATEGORY'])
    df = df.drop(columns=['LINE'], errors='ignore')

    merged = all_line_probs.merge(
        df,
        left_on=['PLAYER_NAME', 'MARKET'],
        right_on=['PLAYER_NAME', 'CATEGORY'],
        how='left',
    ).dropna()

    print(f"[{bookmaker}] {len(merged)} enriched legs")
    return merged

# Category slug → game-log columns (same semantics as dataScraper.cat_to_stat_cols)
_OVER_RATE_CAT_TO_STAT = {
    'player_points': ['PTS'],
    'player_rebounds': ['REB'],
    'player_assists': ['AST'],
    'player_turnovers': ['TOV'],
    'player_frees_attempts': ['FTA'],
    'player_threes': ['FG3M'],
    'player_blocks': ['BLK'],
    'player_steals': ['STL'],
    'player_blocks_steals': ['BLK', 'STL'],
    'player_points_rebounds_assists': ['PTS', 'REB', 'AST'],
    'player_points_rebounds': ['PTS', 'REB'],
    'player_points_assists': ['PTS', 'AST'],
    'player_rebounds_assists': ['REB', 'AST'],
}


def compute_prop_over_rates(
    base_df: pd.DataFrame,
    prop_lines: pd.DataFrame,
    *,
    name_col: str = 'PLAYER_NAME',
    category_col: str = 'CATEGORY',
    line_col: str = 'LINE',
    bookmaker_col: str | None = 'BOOKMAKER',
    line_bookmaker: str | None = None,
    date_col: str = 'GAME_DATE',
) -> pd.DataFrame:
    pl = prop_lines.copy()
    if bookmaker_col and line_bookmaker and bookmaker_col in pl.columns:
        pl = pl[pl[bookmaker_col] == line_bookmaker]
    if pl.empty:
        return pd.DataFrame(
            columns=list(pl.columns)
            + ['OVER_RATE_L5', 'OVER_RATE_L10', 'OVER_RATE_L15', 'OVER_RATE_SEASON']
        )

    pl[line_col] = pd.to_numeric(pl[line_col], errors='coerce')
    pl = pl.dropna(subset=[name_col, category_col, line_col])

    base = base_df.sort_values(date_col)
    rows_out = []

    for _, r in pl.iterrows():
        name = r[name_col]
        cat = r[category_col]
        line = float(r[line_col])
        stat_cols = _OVER_RATE_CAT_TO_STAT.get(cat)
        if not stat_cols:
            rows_out.append({**r.to_dict(), **{
                'OVER_RATE_L5': np.nan,
                'OVER_RATE_L10': np.nan,
                'OVER_RATE_L15': np.nan,
                'OVER_RATE_SEASON': np.nan,
            }})
            continue

        pdf = base[base['PLAYER_NAME'] == name]
        if pdf.empty:
            rows_out.append({**r.to_dict(), **{
                'OVER_RATE_L5': np.nan,
                'OVER_RATE_L10': np.nan,
                'OVER_RATE_L15': np.nan,
                'OVER_RATE_SEASON': np.nan,
            }})
            continue

        missing = [c for c in stat_cols if c not in pdf.columns]
        if missing:
            rows_out.append({**r.to_dict(), **{
                'OVER_RATE_L5': np.nan,
                'OVER_RATE_L10': np.nan,
                'OVER_RATE_L15': np.nan,
                'OVER_RATE_SEASON': np.nan,
            }})
            continue

        if len(stat_cols) == 1:
            s = pdf[stat_cols[0]].astype(float)
        else:
            s = pdf[stat_cols].astype(float).sum(axis=1)

        def _rate(tail_n: int | None) -> float:
            seg = s if tail_n is None else s.tail(tail_n)
            if seg.empty:
                return float('nan')
            return round((seg > line).mean(), 2)

        rows_out.append({
            **r.to_dict(),
            'OVER_RATE_L5': _rate(5),
            'OVER_RATE_L10': _rate(10),
            'OVER_RATE_L15': _rate(15),
            'OVER_RATE_SEASON': _rate(None),
        })

    return pd.DataFrame(rows_out)

from nba_api.stats.endpoints import leaguedashteamstats

# Same as dataScraper: book/log "OPPONENT" vs NBA API team name
_OPP_NAME_TO_API_KEY = {'Los Angeles Clippers': 'LA Clippers'}


def league_team_adv_stats_lookup() -> pd.DataFrame:
    """
    Current league advanced team stats (PerGame, Advanced), one row per team.

    Columns: OPPONENT (team name for merge), OPP_DEF_RATING,
    OPP_RANK_DEF_RATING, OPP_PACE, OPP_PACE_RANK — same naming as
    ``generalized_best_bets`` / ``dataScraper`` after rename.
    """
    league_df = leaguedashteamstats.LeagueDashTeamStats(
        league_id_nullable='00',
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Advanced',
    ).get_data_frames()[0]

    out = (
        league_df[['TEAM_NAME', 'DEF_RATING', 'DEF_RATING_RANK', 'PACE', 'PACE_RANK']]
        .copy()
        .rename(columns={
            'TEAM_NAME': 'OPPONENT',
            'DEF_RATING': 'OPP_DEF_RATING',
            'DEF_RATING_RANK': 'OPP_RANK_DEF_RATING',
            'PACE': 'OPP_PACE',
            'PACE_RANK': 'OPP_PACE_RANK',
        })
    )
    # API uses "LA Clippers"; some pipelines use "Los Angeles Clippers"
    out['_OPP_MATCH_KEY'] = out['OPPONENT'].replace(_OPP_NAME_TO_API_KEY)
    return out


def attach_opponent_adv_stats(
    df: pd.DataFrame,
    opponent_col: str = 'OPPONENT',
    *,
    how: str = 'left',
) -> pd.DataFrame:
    """
    Left-join ``OPP_DEF_RATING``, ``OPP_RANK_DEF_RATING``, ``OPP_PACE``,
    ``OPP_PACE_RANK`` onto ``df`` using ``df[opponent_col]`` (full team name).

    Mirrors the merge in ``dataScraper.py`` (``_output.merge(_opp, ...)``).
    """
    opp = league_team_adv_stats_lookup()
    left = df.copy()
    left['_OPP_MATCH_KEY'] = left[opponent_col].replace(_OPP_NAME_TO_API_KEY)
    merged = left.merge(
        opp[['_OPP_MATCH_KEY', 'OPP_DEF_RATING', 'OPP_RANK_DEF_RATING', 'OPP_PACE', 'OPP_PACE_RANK']],
        on='_OPP_MATCH_KEY',
        how=how,
    ).drop(columns='_OPP_MATCH_KEY')
    return merged


# Same slugs as dataScraper.cat_to_stat_cols (for STAT_VALUE on each game row)
_MATCHUP_CAT_TO_STAT = {
    'player_points': ['PTS'],
    'player_rebounds': ['REB'],
    'player_assists': ['AST'],
    'player_turnovers': ['TOV'],
    'player_frees_attempts': ['FTA'],
    'player_threes': ['FG3M'],
    'player_blocks': ['BLK'],
    'player_steals': ['STL'],
    'player_blocks_steals': ['BLK', 'STL'],
    'player_points_rebounds_assists': ['PTS', 'REB', 'AST'],
    'player_points_rebounds': ['PTS', 'REB'],
    'player_points_assists': ['PTS', 'AST'],
    'player_rebounds_assists': ['REB', 'AST'],
}


def _norm_matchup_opp_key(name: pd.Series) -> pd.Series:
    """Align with dataScraper._norm_matchup_opp (log vs book opponent strings)."""
    s = name.astype(str).str.strip().str.lower()
    return s.replace({'la clippers': 'los angeles clippers'})


def attach_matchup_columns(
    base_df: pd.DataFrame,
    prop_lines: pd.DataFrame,
    *,
    name_col: str = 'PLAYER_NAME',
    category_col: str = 'CATEGORY',
    line_col: str = 'LINE',
    opponent_col_on_props: str = 'OPPONENT',
    opponent_col_in_log: str = 'OPP_OPP_NAME_base',
    date_col: str = 'GAME_DATE',
) -> pd.DataFrame:
    """
    For each prop row, compute historical vs-tonight-opponent stat summary
    (same as ``dataScraper`` ``matchup_agg`` merge + ``MATCHUP_EDGE``):

    - ``AVG_STAT_VS_MATCHUP``: mean ``STAT_VALUE`` in games vs normalized opponent.
    - ``MATCHUP_GAMES``: count of those games.
    - ``MATCHUP_EDGE``: ``AVG_STAT_VS_MATCHUP - LINE``.

    Requires ``base_df[opponent_col_in_log]`` (e.g. ``OPP_OPP_NAME_base``) and
    ``prop_lines[opponent_col_on_props]`` (tonight's opponent label, same
    normalization path as ``merged['OPPONENT']`` in ``generalized_best_bets``).
    """
    if opponent_col_in_log not in base_df.columns:
        out = prop_lines.copy()
        out['AVG_STAT_VS_MATCHUP'] = np.nan
        out['MATCHUP_GAMES'] = np.nan
        out['MATCHUP_EDGE'] = np.nan
        return out

    base = base_df.sort_values(date_col).copy()
    base['_OPP_NORM'] = _norm_matchup_opp_key(base[opponent_col_in_log])

    pl = prop_lines.copy()
    pl[line_col] = pd.to_numeric(pl[line_col], errors='coerce')

    rows = []
    for _, r in pl.iterrows():
        name = r[name_col]
        cat = r[category_col]
        line = r[line_col]
        opp_raw = r.get(opponent_col_on_props, np.nan)

        stat_cols = _MATCHUP_CAT_TO_STAT.get(cat)
        if stat_cols is None or pd.isna(opp_raw):
            rows.append({
                **r.to_dict(),
                'AVG_STAT_VS_MATCHUP': np.nan,
                'MATCHUP_GAMES': np.nan,
                'MATCHUP_EDGE': np.nan,
            })
            continue

        missing = [c for c in stat_cols if c not in base.columns]
        if missing:
            rows.append({
                **r.to_dict(),
                'AVG_STAT_VS_MATCHUP': np.nan,
                'MATCHUP_GAMES': np.nan,
                'MATCHUP_EDGE': np.nan,
            })
            continue

        opp_key = _norm_matchup_opp_key(pd.Series([opp_raw])).iloc[0]
        pdf = base[(base['PLAYER_NAME'] == name) & (base['_OPP_NORM'] == opp_key)]

        if len(stat_cols) == 1:
            sv = pdf[stat_cols[0]].astype(float)
        else:
            sv = pdf[stat_cols].astype(float).sum(axis=1)

        if sv.empty or pd.isna(line):
            avg = np.nan
            n = np.nan
            edge = np.nan
        else:
            avg = round(float(sv.mean()), 2)
            n = int(sv.count())
            edge = round(float(avg - line), 2)

        rows.append({
            **r.to_dict(),
            'AVG_STAT_VS_MATCHUP': avg,
            'MATCHUP_GAMES': n,
            'MATCHUP_EDGE': edge,
        })

    return pd.DataFrame(rows)