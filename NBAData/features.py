import pandas as pd

#grabs players rest days between games
def calculate_days_of_rest(df, player_id_col='PLAYER_ID', game_date_col='GAME_DATE'):
    df[game_date_col] = pd.to_datetime(df[game_date_col], format='%Y-%m-%d')
    df = df.sort_values(by=[player_id_col, game_date_col])
    df['DAYS_OF_REST'] = df.groupby(player_id_col)[game_date_col].diff().dt.days
    return df

#assigns 1 if player is a starter, 0 if not
def starters(data):
    starters = ['G','F','C']
    if data['START_POSITION'] in starters:
        return 1
    else:
        return 0 # use .apply to apply this function to each row or else it wont work

#for points and usage rate
def add_PTS_features(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    # Sort by player and date to ensure rolling works correctly
    player_data = player_data.sort_values([player_id_col, date_col])

    # Add USG_PCT for last 3 and 5 games
    player_data['USG_PCT_LAST_3'] = player_data.groupby(player_id_col)['USG_PCT'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().round(2))
    player_data['USG_PCT_LAST_5'] = player_data.groupby(player_id_col)['USG_PCT'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().round(2))
    player_data['USG_PCT_LAST_7'] = player_data.groupby(player_id_col)['USG_PCT'].transform(lambda x: x.rolling(window=7, min_periods=1).mean().round(2))
    
    # Add average PTS for last 3, 5, and 7 games
    for games in [3, 5, 7]:
        rolling_col_name = f"PTS_LAST_{games}"
        player_data[rolling_col_name] = (
            player_data.groupby(player_id_col)['PTS']
            .transform(lambda x: x.rolling(window=games, min_periods=1).mean().round(2))
        )

    for game in [3, 5, 7]:
        rolling_col_name = f"STD_PTS_LAST_{game}"
        player_data[rolling_col_name] = (
            player_data.groupby(player_id_col)['PTS']
            .transform(lambda x: x.rolling(window=game, min_periods=1).std().round(2))
        )

    # Add home and away average PTS
    for home_away in ['HOME', 'AWAY']:
        avg_column_name = f'PLAYER_{home_away}_AVG_PTS'
        is_home = player_data['HOME_GAME'] == (1 if home_away == 'HOME' else 0)
        player_data[avg_column_name] = (
        player_data.groupby(player_id_col)
        .apply(lambda group: group.loc[is_home, 'PTS'].expanding().mean().round(2), include_groups=False)
        .reset_index(level=0, drop=True)
    )

    return player_data

#rolling averages for points against each team
def add_matchup_points(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION'):
    player_data = player_data.sort_values([player_id_col, 'GAME_DATE'])
    
    # Calculate recent average points (last 3 games)
    player_data['MATCHUP_AVG_PTS_LAST_3'] = (
        player_data.groupby([player_id_col, opp_col])['PTS']
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean().round(2))
    )
    
    # Count number of games against this team
    player_data['GAMES_VS_OPP'] = player_data.groupby([player_id_col, opp_col]).cumcount() + 1
    
    return player_data