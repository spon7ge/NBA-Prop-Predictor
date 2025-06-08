def starters(data):
    starters = ['G','F','C']
    if data['START_POSITION'] in starters:
        return 1
    else:
        return 0
    
def assign_position(data):
    unique_ids = data['PLAYER_ID'].unique()
    ids = {}
    
    for i in unique_ids:
        try:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=i).get_data_frames()[0]
            print(f"Gathered data for PLAYER_ID {i}...")
            if not player_info.empty:
                ids[i] = player_info.iloc[0]['POSITION']
        except Exception as e:
            print(f"Error fetching data for PLAYER_ID {i}: {e}")
            ids[i] = None
        time.sleep(1)
        
    data['POSITION'] = data['PLAYER_ID'].map(ids)
    
pos_dummies = pd.get_dummies(data['POSITION'], prefix='')
team_dummies = pd.get_dummies(data['TEAM_ABBREVIATION'], prefix='TEAM').astype(int)
opp_dummies = pd.get_dummies(data['OPP_ABBREVIATION'], prefix='OPP').astype(int)