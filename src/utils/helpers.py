"""
Data-loading and bookmaker-merge helpers for the daily pipeline.
All functions are pure (no side-effects on import).
"""
import json
import joblib
import pandas as pd
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
) -> pd.DataFrame:
    """
    Run generalized_best_bets for one bookmaker and left-join the enriched
    contextual stats onto the model probabilities in all_line_probs.

    Returns an enriched DataFrame ready for build_greedy_slate(), or an empty
    DataFrame if no lines were found for the bookmaker today.
    """
    _, _, final = generalized_best_bets(
        lines_dfs, base_df, lines_us, team_odds,
        line_bookmaker=bookmaker,
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
