import numpy as np
import pandas as pd
from scipy import stats
from nba_api.stats.endpoints import leaguedashteamstats

prop_categories = [
    'player_points',
    'player_rebounds',
    'player_assists',
    'player_turnovers',
    'player_frees_attempts',
    'player_threes',
    'player_blocks',
    'player_steals',
    'player_blocks_steals',
    'player_points_rebounds_assists',
    'player_points_rebounds',
    'player_points_assists',
    'player_rebounds_assists',
]

cat_to_stat_cols = {
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


def implied_prob(american_odds: float) -> float:
    if american_odds > 0:
        return round(100 / (american_odds + 100), 3)
    return round(abs(american_odds) / (abs(american_odds) + 100), 3)


def calc_ev(prob: float, american_odds: float) -> float:
    decimal = (american_odds / 100 + 1) if american_odds > 0 else (100 / abs(american_odds) + 1)
    return round(((prob * (decimal - 1)) - (1 - prob)) * 100, 2)


def _norm_matchup_opp(name: pd.Series) -> pd.Series:
    """Map alternate book/log spellings to one key (e.g. LA Clippers vs Los Angeles Clippers)."""
    s = name.astype(str).str.strip().str.lower()
    return s.replace({'la clippers': 'los angeles clippers'})


def generalized_best_bets(
    lines_dfs,
    base_df,
    us_df,
    team_dds,
    line_bookmaker='Underdog',
    game_odds_df=None,
):
    """Prop lines from ``lines_dfs`` for ``line_bookmaker``; best US sides from ``us_df``."""
    if game_odds_df is None:
        game_rows = []

        for game in team_dds.to_dict('records'):
            home = game['home_team']
            away = game['away_team']
            commence = game['commence_time']
            bookmakers = game['bookmakers']

            spreads_home, spreads_away, totals = [], [], []

            for bk in bookmakers:
                for market in bk['markets']:
                    if market['market_key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home:
                                spreads_home.append(outcome['point'])
                            elif outcome['name'] == away:
                                spreads_away.append(outcome['point'])
                    elif market['market_key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':  # one side is enough
                                totals.append(outcome['point'])

            consensus_total = round(np.median(totals), 1) if totals else None
            consensus_spread_home = round(np.median(spreads_home), 1) if spreads_home else None
            consensus_spread_away = round(np.median(spreads_away), 1) if spreads_away else None
            n_books = len(bookmakers)

            game_rows.append({
                'TEAM': home,
                'OPPONENT': away,
                'TEAM_SPREAD': consensus_spread_home,
                'GAME_TOTAL': consensus_total,
                'N_BOOKS': n_books,
                'COMMENCE_TIME': commence,
                'HOME_AWAY': 'HOME',
            })
            game_rows.append({
                'TEAM': away,
                'OPPONENT': home,
                'TEAM_SPREAD': consensus_spread_away,
                'GAME_TOTAL': consensus_total,
                'N_BOOKS': n_books,
                'COMMENCE_TIME': commence,
                'HOME_AWAY': 'AWAY',
            })

        game_odds_df = pd.DataFrame(game_rows)

    outputs = []

    for category in prop_categories:
        stat_cols = cat_to_stat_cols[category]

        book_cat_mask = (lines_dfs['BOOKMAKER'] == line_bookmaker) & (lines_dfs['CATEGORY'] == category)
        cat_line_names = lines_dfs.loc[book_cat_mask, 'NAME'].unique()
        df = base_df[base_df['PLAYER_NAME'].isin(cat_line_names)].copy()

        if df.empty:
            continue

        if len(stat_cols) == 1:
            df['STAT_VALUE'] = df[stat_cols[0]]
        else:
            df['STAT_VALUE'] = df[stat_cols].sum(axis=1)

        df['AVG_MIN_L10'] = df.groupby('PLAYER_ID')['MIN'].transform(lambda x: x.rolling(10).mean().round(2))
        df['STD_MIN_L10'] = df.groupby('PLAYER_ID')['MIN'].transform(lambda x: x.rolling(10).std().round(2))
        df['AVG_USG_L10'] = df.groupby('PLAYER_ID')['USG_PCT'].transform(lambda x: x.rolling(10).mean().round(2))
        df['STD_USG_L10'] = df.groupby('PLAYER_ID')['USG_PCT'].transform(lambda x: x.rolling(10).std().round(2))

        df['AVG_STAT_L10'] = df.groupby('PLAYER_ID')['STAT_VALUE'].transform(lambda x: x.rolling(10).mean().round(2))
        df['STD_STAT_L10'] = df.groupby('PLAYER_ID')['STAT_VALUE'].transform(lambda x: x.rolling(10).std().round(2))
        df['MED_STAT_L10'] = df.groupby('PLAYER_ID')['STAT_VALUE'].transform(lambda x: x.rolling(10).median().round(2))

        latest = df.groupby('PLAYER_ID').last().reset_index()

        prop_lines = (
            lines_dfs.loc[book_cat_mask, ['NAME', 'LINE', 'ODDS', 'COMMENCE_TIME']]
            .rename(columns={'NAME': 'PLAYER_NAME'})
            .copy()
        )

        if prop_lines.empty:
            continue

        prop_lines['LINE'] = prop_lines['LINE'].astype(float)
        prop_lines = prop_lines.drop_duplicates('PLAYER_NAME')

        merged = latest.merge(prop_lines, on='PLAYER_NAME', how='inner')
        # S26 uses `Position`; downstream columns expect `POSITION` (per player, from latest row).
        if 'Position' in merged.columns:
            merged['POSITION'] = merged['Position']
        elif 'POSITION' not in merged.columns:
            merged['POSITION'] = np.nan
        merged['CATEGORY'] = category
        merged['LINE_BOOKMAKER'] = line_bookmaker

        real_odds = us_df[us_df['CATEGORY'] == category].rename(columns={'NAME': 'PLAYER_NAME'}).copy()
        real_odds['LINE'] = real_odds['LINE'].astype(float)

        for side, col in [('Over', 'ODDS_OVER'), ('Under', 'ODDS_UNDER')]:
            best = (
                real_odds[real_odds['OVER/UNDER'] == side]
                .groupby(['PLAYER_NAME', 'LINE'])['ODDS'].max()
                .reset_index()
                .rename(columns={'ODDS': col})
            )
            merged = merged.merge(best, on=['PLAYER_NAME', 'LINE'], how='left')

        merged['ODDS_OVER'] = merged['ODDS_OVER'].fillna(-137).astype(int)
        merged['ODDS_UNDER'] = merged['ODDS_UNDER'].fillna(-137).astype(int)

        merged = merged.merge(
            game_odds_df[['TEAM', 'OPPONENT', 'TEAM_SPREAD', 'GAME_TOTAL', 'N_BOOKS', 'HOME_AWAY']],
            left_on='TEAM_NAME',
            right_on='TEAM',
            how='left'
        ).drop(columns='TEAM')

        if 'OPP_OPP_NAME_base' in df.columns:
            _hist = df.dropna(subset=['OPP_OPP_NAME_base']).copy()
            _hist['_OPP_NORM'] = _norm_matchup_opp(_hist['OPP_OPP_NAME_base'])
            merged['_OPP_NORM'] = _norm_matchup_opp(merged['OPPONENT'])
            matchup_agg = (
                _hist.groupby(['PLAYER_NAME', '_OPP_NORM'], as_index=False)
                .agg(AVG_STAT_VS_MATCHUP=('STAT_VALUE', 'mean'), MATCHUP_GAMES=('STAT_VALUE', 'count'))
            )
            matchup_agg['AVG_STAT_VS_MATCHUP'] = matchup_agg['AVG_STAT_VS_MATCHUP'].round(2)
            merged = merged.merge(matchup_agg, on=['PLAYER_NAME', '_OPP_NORM'], how='left')
            merged = merged.drop(columns=['_OPP_NORM'])
            merged['MATCHUP_EDGE'] = (merged['AVG_STAT_VS_MATCHUP'] - merged['LINE']).round(2)
        else:
            merged['AVG_STAT_VS_MATCHUP'] = np.nan
            merged['MATCHUP_GAMES'] = np.nan
            merged['MATCHUP_EDGE'] = np.nan

        merged['IMP_PROB_OVER'] = merged['ODDS_OVER'].apply(implied_prob)
        merged['IMP_PROB_UNDER'] = merged['ODDS_UNDER'].apply(implied_prob)

        merged['EDGE'] = (merged['AVG_STAT_L10'] - merged['LINE']).round(2)
        merged['MED_EDGE'] = (merged['MED_STAT_L10'] - merged['LINE']).round(2)
        z_denom = merged['STD_STAT_L10'].replace(0, np.nan)
        merged['Z_SCORE'] = ((merged['LINE'] - merged['AVG_STAT_L10']) / z_denom).round(3)
        merged['PROB_OVER'] = (1 - stats.norm.cdf(merged['Z_SCORE'])).round(3)
        merged['PROB_UNDER'] = stats.norm.cdf(merged['Z_SCORE']).round(3)

        merged['TOTAL_BOOST'] = ((merged['GAME_TOTAL'] - 220) / 10).round(3)
        merged['IS_UNDERDOG'] = (merged['TEAM_SPREAD'] > 0).astype(int)

        cover_df = df.merge(merged[['PLAYER_NAME', 'LINE']], on='PLAYER_NAME', how='inner')

        cover = cover_df.groupby('PLAYER_NAME').apply(
            lambda g: pd.Series({
                'OVER_RATE_L5': (g['STAT_VALUE'].tail(5) > g['LINE'].iloc[0]).mean().round(2),
                'OVER_RATE_L10': (g['STAT_VALUE'].tail(10) > g['LINE'].iloc[0]).mean().round(2),
                'OVER_RATE_L15': (g['STAT_VALUE'].tail(15) > g['LINE'].iloc[0]).mean().round(2),
                'OVER_RATE_SEASON': (g['STAT_VALUE'] > g['LINE'].iloc[0]).mean().round(2),
            })
        ).reset_index()

        merged = merged.merge(cover, on='PLAYER_NAME', how='left')

        merged['EV_OVER'] = merged.apply(lambda r: calc_ev(r['PROB_OVER'], r['ODDS_OVER']), axis=1)
        merged['EV_UNDER'] = merged.apply(lambda r: calc_ev(r['PROB_UNDER'], r['ODDS_UNDER']), axis=1)

        merged = merged[(merged['AVG_MIN_L10'] >= 20) & (merged['STD_MIN_L10'] <= 8)].copy()

        merged['BET_FLAG'] = (
            (merged['PROB_OVER'] >= 0.60)
            & (merged['EV_OVER'] > 0)
        )

        output_cat = merged[[
            'PLAYER_NAME',
            'TEAM_NAME',
            'OPPONENT',
            'HOME_AWAY',
            'TEAM_SPREAD',
            'GAME_TOTAL',
            'CATEGORY',
            'LINE_BOOKMAKER',
            'LINE',
            'ODDS_OVER',
            'ODDS_UNDER',
            'IMP_PROB_OVER',
            'IMP_PROB_UNDER',
            'AVG_STAT_L10',
            'MED_STAT_L10',
            'AVG_STAT_VS_MATCHUP',
            'MATCHUP_GAMES',
            'MATCHUP_EDGE',
            'STD_STAT_L10',
            'EDGE',
            'MED_EDGE',
            'Z_SCORE',
            'PROB_OVER',
            'PROB_UNDER',
            'EV_OVER',
            'EV_UNDER',
            'OVER_RATE_L5',
            'OVER_RATE_L10',
            'OVER_RATE_L15',
            'OVER_RATE_SEASON',
            'AVG_MIN_L10',
            'STD_MIN_L10',
            'AVG_USG_L10',
            'STD_USG_L10',
            'TOTAL_BOOST',
            'IS_UNDERDOG',
            'BET_FLAG',
            'COMMENCE_TIME',
        ]].sort_values('EV_OVER', ascending=False)

        outputs.append(output_cat)

    output_all = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    tier1_all = output_all[output_all['BET_FLAG']] if not output_all.empty else output_all

    if output_all.empty:
        final = pd.DataFrame()
        return output_all, tier1_all, final

    league_df = leaguedashteamstats.LeagueDashTeamStats(
        league_id_nullable='00',
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Advanced',
    ).get_data_frames()[0]

    opp_stats = (
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

    _opp_lookup_map = {'Los Angeles Clippers': 'LA Clippers'}
    _output = output_all.assign(
        _OPP_MATCH_KEY=output_all['OPPONENT'].replace(_opp_lookup_map)
    )
    _opp = opp_stats.rename(columns={'OPPONENT': '_OPP_MATCH_KEY'})
    final = _output.merge(_opp, on='_OPP_MATCH_KEY', how='left').drop(columns='_OPP_MATCH_KEY')

    final = final[[
        'PLAYER_NAME', 'LINE', 'CATEGORY', 'LINE_BOOKMAKER', 'OPPONENT',
        'TEAM_SPREAD', 'GAME_TOTAL', 'OPP_DEF_RATING', 'OPP_RANK_DEF_RATING', 'OPP_PACE', 'OPP_PACE_RANK',
        'ODDS_OVER', 'ODDS_UNDER',
        'IMP_PROB_OVER', 'IMP_PROB_UNDER',
        'AVG_STAT_L10', 'MED_STAT_L10', 'STD_STAT_L10', 'EDGE', 'MED_EDGE', 'Z_SCORE',
        'PROB_OVER', 'PROB_UNDER', 'EV_OVER', 'EV_UNDER',
        'OVER_RATE_L5', 'OVER_RATE_L10', 'OVER_RATE_L15', 'OVER_RATE_SEASON',
        'AVG_MIN_L10', 'STD_MIN_L10', 'AVG_USG_L10', 'STD_USG_L10', 'AVG_STAT_VS_MATCHUP', 'MATCHUP_GAMES',
    ]].sort_values('EV_OVER', ascending=False)

    return output_all, tier1_all, final
