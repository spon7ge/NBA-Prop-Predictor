import pandas as pd
import numpy as np

def ppm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    df = _rolling_player(df)
    df = _ewm_player(df)
    df = _lag_features(df)
    df = _starter_features(df)
    df = _season_averages(df)
    df = _team_context(df)
    df = _schedule_features(df)
    df = _volatility_features(df)
    df = _opponent_stats(df)
    df = _expectedPace(df)
    df = _detect_star_players(df)
    df = _compute_ppm_by_active_count(df)

    return df

def _resolve_position_column(df: pd.DataFrame):
    """Resolve position column; prefers ``pos`` (matches ``Pos`` / ``POS`` by name)."""
    if 'pos' in df.columns:
        return 'pos'
    lower_to_orig = {str(c).lower(): c for c in df.columns}
    if 'pos' in lower_to_orig:
        return lower_to_orig['pos']
    for col in ('POSITION', 'START_POSITION'):
        if col in df.columns:
            return col
    return None


def _normalize_pos_series(s: pd.Series) -> pd.Series:
    """String compare helper: strip and casefold for robust equality."""
    out = s.astype('string')
    return out.str.strip().str.casefold()


def _resolve_star_team_col(df: pd.DataFrame) -> str:
    """Column for franchise / roster side in star logic (same team within ``GAME_ID``)."""
    if 'TEAM_ID' in df.columns:
        return 'TEAM_ID'
    if 'TEAM_ABBREVIATION' in df.columns:
        return 'TEAM_ABBREVIATION'
    raise KeyError(
        "Star and redistribution features require 'TEAM_ID' or 'TEAM_ABBREVIATION' in the dataframe"
    )


def _position_prior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prior-game position per player (shift within PLAYER_ID timeline).
    Avoids using the current game's listed position when labeling historical rows.
    """
    df = df.copy()
    pos_col = _resolve_position_column(df)
    if pos_col is None:
        df['POS_PRIOR'] = pd.NA
        return df
    df['POS_PRIOR'] = df.groupby('PLAYER_ID', sort=False)[pos_col].shift(1)
    return df

# ── 1. Rolling player performance ─────────────────────────────────────────────

def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'USG_PCT', 'PLUS_MINUS', 'POSS', 'PTS_PER_MIN', 'PTS_PER_POSS',
    'TS_PCT', 'AST_TO', 'FGA_PER_MIN', '3PA_PER_MIN', 'FTA_PER_MIN', 'AST', 'TOV'
    ]

    for window in [3,5,10]:
        for col in cols:
            df[f'{col}_roll{window}'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean().round(2))
            )

    return df

def _ewm_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'MIN', 'PTS', 'USG_PCT', 'PLUS_MINUS', 'POSS', 'PTS_PER_MIN', 'PTS_PER_POSS', 'POSS_PER_MIN',
        'TS_PCT', 'AST_TO', 'FGA_PER_MIN', '3PA_PER_MIN', 'FTA_PER_MIN', 'AST', 'TOV',
    ]

    for span in [3, 5, 10]:
        for col in cols:
            df[f'{col}_{span}_ewm'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(
                    lambda x: x.shift(1).ewm(span=span, adjust=False).mean().round(2)
                )
            )

    return df

# ── 2. Lag features ───────────────────────────────────────────────────────────

def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'PLUS_MINUS', 'PIE', 'STARTING', 'POSS', "USG_PCT", "PTS_PER_MIN", '3PA_PER_MIN', 'FTA_PER_MIN', 'POSS_PER_MIN']

    for lag in [1, 2]:
        for col in cols:
            df[f'{col}_lag{lag}'] = df.groupby('PLAYER_ID')[col].shift(lag)

    return df


# ── 3. Starter / role ─────────────────────────────────────────────────────────

def _starter_features(df: pd.DataFrame) -> pd.DataFrame:
    df['STARTER_ROLL10_PCT'] = (
        df.groupby('PLAYER_ID')['STARTING']
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean().round(2))
    )
    return df


# ── 4. Season averages (expanding) ───────────────────────────────────────────

def _season_averages(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'USG_PCT', 'POSS', 'PF', 'OFF_RATING', 'DEF_RATING', 'PTS_PER_MIN', 'FGA_PER_MIN', 'FTA_PER_MIN', '3PA_PER_MIN', 'TS_PCT']

    for col in cols:
        df[f'{col}_season_avg'] = (
            df.groupby(['PLAYER_ID', 'SEASON_YEAR'])[col]
            .transform(lambda x: x.shift(1).expanding().mean().round(2))
        )

    return df

# ── 6. Team context ───────────────────────────────────────────────────────────

def _team_context(df: pd.DataFrame) -> pd.DataFrame:
    # ── Team pace: one row per team per game, then map back ──────────────────
    team_pace = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_PACE']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )
    team_pts = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_PTS']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )
    team_poss = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_POSS']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )
    team_ts_pct = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_TS_PCT']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )

    team_pace_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_PACE_roll10=team_pace.values)[['TEAM_ID', 'GAME_ID', 'TEAM_PACE_roll10']]
    )
    team_pts_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_PTS_roll10=team_pts.values)[['TEAM_ID', 'GAME_ID', 'TEAM_PTS_roll10']]
    )
    team_poss_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_POSS_roll10=team_poss.values)[['TEAM_ID', 'GAME_ID', 'TEAM_POSS_roll10']]
    )
    team_ts_pct_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_TS_PCT_roll10=team_ts_pct.values)[['TEAM_ID', 'GAME_ID', 'TEAM_TS_PCT_roll10']]
    )

    df = df.merge(team_pace_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_pts_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_poss_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_ts_pct_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    # ── MIN share and POSS share are player-level so these are fine as-is ─────
    df["MIN_share_proxy"] = round(df["MIN_roll10"] / (48 * 10), 2)
    #percentage of team's points scored by the player
    df["PTS_share_proxy"] = round(df["PTS_roll10"] / (df["TEAM_PTS_roll10"]), 2)

    # df["TEAM_POSS_roll5"] = df.groupby(["TEAM_ID", "SEASON_YEAR"])["POSS"].transform(lambda x: x.shift(1).rolling(5).mean().round(2))
    # df["TEAM_POSS_share"] = df["POSS_roll5"] / (df["TEAM_POSS_roll5"] + 1e-6)

    return df

# ── 8. Schedule / rest ────────────────────────────────────────────────────────

def _schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    df['DAYS_REST'] = (
        df.groupby('PLAYER_ID')['GAME_DATE']
        .diff().dt.days.fillna(3)
    )

    df['IS_B2B']  = (df['DAYS_REST'] == 1).astype(int)
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs', na=False).astype(int)

    df['GAME_NUMBER'] = (
        df.groupby(['PLAYER_ID', 'SEASON_YEAR'])
        .cumcount()
    )

    return df


# ── 9. Volatility ─────────────────────────────────────────────────────────────

def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df['MIN_std10'] = (
        df.groupby('PLAYER_ID')['MIN']
        .transform(lambda x: x.shift(1).rolling(10).std().round(2))
    )

    return df
# ---- Opponent stats ---------------------------------------------------------------
def _opponent_stats(df):
    stat_cols = ['TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_POSS']
    available_stat_cols = [c for c in stat_cols if c in df.columns]

    team_game = (
        df[['GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION'] + available_stat_cols]
        .drop_duplicates(subset=['GAME_ID', 'TEAM_ABBREVIATION'])
        .sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    )

    opp_cols = []
    for col in available_stat_cols:
        base = col.replace('TEAM_', '')
        for window in [5, 10]:
            avg_col = f'{base}_roll{window}'
            team_game[avg_col] = (
                team_game.groupby('TEAM_ABBREVIATION')[col]
                .transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean().round(2))
            )
            opp_cols.append(avg_col)

    opp_rename = {col: f'OPP_{col}' for col in opp_cols}

    team_game_opp = (
        team_game[['GAME_ID', 'TEAM_ABBREVIATION'] + opp_cols]
        .rename(columns={**{'TEAM_ABBREVIATION': 'OPP_OPP_ABBREVIATION_base'}, **opp_rename})
    )

    return df.merge(team_game_opp, on=['GAME_ID', 'OPP_OPP_ABBREVIATION_base'], how='left')

# ---- Expected Pace and Points -----------------------
def _expectedPace(df):
    df = df.copy()
    df['EXPECTED_PACE'] = ((df['TEAM_PACE_roll10'] + df['OPP_PACE_roll10']) / 2).round(2)
    df['PACE_DIFFERENTIAL'] = df['TEAM_PACE_roll10'] - df['OPP_PACE_roll10']    
    return df

# ---- Finding Star Players -----------------------
def _detect_star_players(df, min_minutes=10, min_games=10, name_dict=None):
    df = df.copy()
    # ppm_features() does not call :func:`_position_prior` earlier (min_features does); ensure prior pos exists.
    df = _position_prior(df)

    # 1. Normalize Names
    if name_dict:
        df['PLAYER_NAME_NORM'] = df['PLAYER_NAME'].map(lambda x: name_dict.get(x, x))
    else:
        df['PLAYER_NAME_NORM'] = df['PLAYER_NAME']

    team_col = _resolve_star_team_col(df)

    # 2–5. Causal star ranks: prior-only expanding stats per player, min–max within each
    #      (GAME_ID, team), then rank teammates for that game (no season-wide leakage).
    df['_sort_order'] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values([team_col, 'PLAYER_NAME_NORM', 'GAME_DATE'], kind='mergesort')

    grp_keys = [df[team_col], df['PLAYER_NAME_NORM']]
    qual = df['MIN'] >= min_minutes
    df['_prior_qual_games'] = (
        qual.astype(int).groupby(grp_keys, sort=False).transform(lambda s: s.shift(1).cumsum())
    )

    cols = ['USG_PCT', 'TS_PCT', 'EFG_PCT', 'PTS', 'PIE', 'NET_RATING']
    prior_cols = []
    for col in cols:
        pc = f'_prior_{col}'
        if col not in df.columns:
            df[pc] = np.nan
        else:
            v = pd.to_numeric(df[col], errors='coerce').where(qual)
            df[pc] = v.groupby(grp_keys, sort=False).transform(
                lambda s: s.shift(1).expanding().mean()
            )
        prior_cols.append(pc)

    def _norm_team_game(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors='coerce').astype(float)
        mn = s.min(skipna=True)
        mx = s.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or mx <= mn:
            return pd.Series(0.0, index=s.index, dtype=float)
        return ((s - mn) / (mx - mn)).fillna(0.0)

    norm_cols = []
    for pc in prior_cols:
        nc = f'{pc}_TG_NORM'
        df[nc] = df.groupby(['GAME_ID', team_col], sort=False)[pc].transform(_norm_team_game)
        norm_cols.append(nc)

    # Same weights as legacy season-aggregate score (EFG/NET normalized but not in weighted sum)
    df['_star_score_ranking'] = (
        0.30 * df['_prior_USG_PCT_TG_NORM']
        + 0.25 * df['_prior_TS_PCT_TG_NORM']
        + 0.30 * df['_prior_PTS_TG_NORM']
        + 0.15 * df['_prior_PIE_TG_NORM']
    )
    eligible = df['_prior_qual_games'] >= min_games
    df['_star_score_ranking'] = df['_star_score_ranking'].where(eligible)

    df['STAR_RANK'] = (
        df.groupby(['GAME_ID', team_col], sort=False)['_star_score_ranking']
        .rank(method='first', ascending=False, na_option='bottom')
    )
    df.loc[(df['STAR_RANK'] > 3) | (~eligible), 'STAR_RANK'] = np.nan

    drop_tmp = ['_sort_order', '_prior_qual_games', '_star_score_ranking'] + prior_cols + norm_cols
    df = df.sort_values('_sort_order', kind='mergesort').drop(columns=drop_tmp, errors='ignore')

    rank_to_col = {
        1: 'MAIN_STAR_ACTIVE',
        2: 'SECONDARY_STAR_ACTIVE',
        3: 'THIRD_STAR_ACTIVE',
    }

    sr_valid = df['STAR_RANK'].notna()
    df['IS_STAR_TRIO'] = sr_valid.astype(int)
    # Rank-specific flags (IS_TOP_STAR = slot 1 / "top" star only)
    df['IS_TOP_STAR'] = (df['STAR_RANK'] == 1).astype(int)
    df['IS_SECOND_STAR'] = (df['STAR_RANK'] == 2).astype(int)
    df['IS_THIRD_STAR'] = (df['STAR_RANK'] == 3).astype(int)

    # 6. For each (GAME_ID, team), figure out which of the team's top-3 stars actually played
    star_rows = df.loc[sr_valid].copy()
    star_rows['_PLAYED'] = (star_rows['MIN'] >= min_minutes).astype(int)

    active_per_game = (
        star_rows
        .pivot_table(
            index=['GAME_ID', team_col],
            columns='STAR_RANK',
            values='_PLAYED',
            aggfunc='max',
            fill_value=0,
        )
        .rename(columns=rank_to_col)
        .reset_index()
    )
    active_per_game.columns.name = None

    # Make sure all three flag columns exist even if a tier never appears
    for col in rank_to_col.values():
        if col not in active_per_game.columns:
            active_per_game[col] = 0

    flag_cols = list(rank_to_col.values())

    # Drop pre-existing copies to keep merges clean on re-runs
    df = df.drop(columns=[c for c in flag_cols if c in df.columns], errors='ignore')

    df = df.merge(
        active_per_game[['GAME_ID', team_col] + flag_cols],
        on=['GAME_ID', team_col],
        how='left',
    )
    for col in flag_cols:
        df[col] = df[col].fillna(0).astype(int)

    # 7. Star slot positions (from each star's prior-game position) + positional match flags
    star_pos_cols = ['STAR_1_POS', 'STAR_2_POS', 'STAR_3_POS']
    match_cols = ['STAR_1_POSITION_MATCH', 'STAR_2_POSITION_MATCH', 'STAR_3_POSITION_MATCH']
    df = df.drop(columns=[c for c in star_pos_cols + match_cols if c in df.columns], errors='ignore')

    sr = star_rows.dropna(subset=['STAR_RANK']).copy()
    pos_col = _resolve_position_column(df)
    if not sr.empty:
        slot_src = sr['POS_PRIOR']
        if pos_col is not None:
            slot_src = sr['POS_PRIOR'].combine_first(sr[pos_col])
        slot_pos = (
            sr.assign(_SLOT_POS=slot_src)
            .groupby(['GAME_ID', team_col, 'STAR_RANK'], as_index=False)['_SLOT_POS']
            .first()
        )
        wide = slot_pos.pivot(index=['GAME_ID', team_col], columns='STAR_RANK', values='_SLOT_POS')
        wide = wide.rename(columns={c: f'STAR_{int(float(c))}_POS' for c in wide.columns})
        df = df.merge(wide.reset_index(), on=['GAME_ID', team_col], how='left')
        for c in star_pos_cols:
            if c not in df.columns:
                df[c] = pd.NA

        player_pos = df['POS_PRIOR']
        if pos_col is not None:
            player_pos = df['POS_PRIOR'].combine_first(df[pos_col])
        pp = _normalize_pos_series(player_pos)
        for col_pos, col_match in (
            ('STAR_1_POS', 'STAR_1_POSITION_MATCH'),
            ('STAR_2_POS', 'STAR_2_POSITION_MATCH'),
            ('STAR_3_POS', 'STAR_3_POSITION_MATCH'),
        ):
            sp = _normalize_pos_series(df[col_pos])
            df[col_match] = (
                pp.notna() & sp.notna() & (pp == sp)
            ).astype(int)
    else:
        for col_pos, col_match in zip(star_pos_cols, match_cols):
            df[col_pos] = pd.NA
            df[col_match] = 0

    return df.drop(columns=['PLAYER_NAME_NORM'])

def _compute_ppm_by_active_count(df, min_minutes=10):
    df = df.copy()
    team_col = _resolve_star_team_col(df)

    # Per (GAME_ID, team): star slot PPM played (``*_ACTIVE`` comes from :func:`_detect_star_players`).
    _redist_cols = (
        'MAIN_STAR_PPM_PLAYED',
        'SECONDARY_STAR_PPM_PLAYED',
        'THIRD_STAR_PPM_PLAYED',
    )
    df = df.drop(columns=[c for c in _redist_cols if c in df.columns], errors='ignore')

    if 'STAR_RANK' in df.columns:
        slots = df.loc[df['STAR_RANK'].notna(), ['GAME_ID', team_col, 'STAR_RANK', 'PTS_PER_MIN']].copy()
        slots['PTS_PER_MIN'] = pd.to_numeric(slots['PTS_PER_MIN'], errors='coerce').fillna(0)
        slot_ppm = slots.groupby(['GAME_ID', team_col, 'STAR_RANK'], as_index=False)['PTS_PER_MIN'].max()
        if not slot_ppm.empty:
            wide = slot_ppm.pivot(index=['GAME_ID', team_col], columns='STAR_RANK', values='PTS_PER_MIN')
            wide.columns = [int(float(c)) for c in wide.columns]
            for rk in (1, 2, 3):
                if rk not in wide.columns:
                    wide[rk] = 0.0
            wide = wide.rename(
                columns={
                    1: 'MAIN_STAR_PPM_PLAYED',
                    2: 'SECONDARY_STAR_PPM_PLAYED',
                    3: 'THIRD_STAR_PPM_PLAYED',
                }
            ).reset_index()
            df = df.merge(wide, on=['GAME_ID', team_col], how='left')
        else:
            for c in ('MAIN_STAR_PPM_PLAYED', 'SECONDARY_STAR_PPM_PLAYED', 'THIRD_STAR_PPM_PLAYED'):
                df[c] = 0.0
    else:
        for c in ('MAIN_STAR_PPM_PLAYED', 'SECONDARY_STAR_PPM_PLAYED', 'THIRD_STAR_PPM_PLAYED'):
            df[c] = 0.0

    for c in ('MAIN_STAR_PPM_PLAYED', 'SECONDARY_STAR_PPM_PLAYED', 'THIRD_STAR_PPM_PLAYED'):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0)

    tier_cols = {
        'MAIN_STAR_ACTIVE': 'main_star',
        'SECONDARY_STAR_ACTIVE': 'secondary_star',
        'THIRD_STAR_ACTIVE': 'third_star',
    }

    # ── Causal per-row expanding means of PPM over PRIOR games (no leakage) ────
    new_cols = [
        'ppm_main_star_active', 'ppm_main_star_out', 'ppm_main_star_jump_when_out',
        'ppm_secondary_star_active', 'ppm_secondary_star_out', 'ppm_secondary_star_jump_when_out',
        'ppm_third_star_active', 'ppm_third_star_out', 'ppm_third_star_jump_when_out',
        'ppm_all_stars_out',
    ]
    df = df.drop(columns=[c for c in new_cols if c in df.columns], errors='ignore')

    if 'PTS_PER_MIN' not in df.columns:
        _min = pd.to_numeric(df['MIN'], errors='coerce').replace(0, np.nan)
        df['PTS_PER_MIN'] = (pd.to_numeric(df['PTS'], errors='coerce') / _min).round(4)

    original_index = df.index.copy()
    df = df.sort_values(['PLAYER_NAME', team_col, 'GAME_DATE']).copy()

    _min_played = pd.to_numeric(df['MIN'], errors='coerce')
    played_mask = _min_played >= min_minutes
    keys = [df['PLAYER_NAME'], df[team_col]]
    ppm_vals = pd.to_numeric(df['PTS_PER_MIN'], errors='coerce').fillna(0.0)

    def _prior_mean_ppm(value_mask: pd.Series) -> pd.Series:
        """Expanding mean of PTS_PER_MIN over prior rows where value_mask is True (per group)."""
        masked = ppm_vals.where(value_mask, 0)
        masked_cnt = value_mask.astype(int)
        cs = masked.groupby(keys, sort=False).cumsum()
        cc = masked_cnt.groupby(keys, sort=False).cumsum()
        prior_sum = cs - masked
        prior_cnt = cc - masked_cnt
        return (prior_sum / prior_cnt.where(prior_cnt > 0)).round(4)

    for flag_col, tier_label in tier_cols.items():
        active_col = f'ppm_{tier_label}_active'
        out_col = f'ppm_{tier_label}_out'
        if flag_col not in df.columns:
            df[active_col] = float('nan')
            df[out_col] = float('nan')
            df[f'ppm_{tier_label}_jump_when_out'] = float('nan')
            continue
        df[active_col] = _prior_mean_ppm(played_mask & (df[flag_col] == 1))
        df[out_col] = _prior_mean_ppm(played_mask & (df[flag_col] == 0))
        jump_col = f'ppm_{tier_label}_jump_when_out'
        df[jump_col] = (
            pd.to_numeric(df[out_col], errors='coerce')
            - pd.to_numeric(df[active_col], errors='coerce')
        ).round(4)

    flag_set = ['MAIN_STAR_ACTIVE', 'SECONDARY_STAR_ACTIVE', 'THIRD_STAR_ACTIVE']
    if all(c in df.columns for c in flag_set):
        all_out = (df[flag_set] == 0).all(axis=1)
        df['ppm_all_stars_out'] = _prior_mean_ppm(played_mask & all_out)
    else:
        df['ppm_all_stars_out'] = float('nan')

    df = df.loc[original_index]
    return df
