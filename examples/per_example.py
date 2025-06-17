import pandas as pd
import numpy as np
from NBAData.features import CalculatePER

def analyze_player_per(df, player_name=None):
    """
    Analyze PER statistics for all players or a specific player
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing player game statistics
    player_name : str, optional
        Specific player name to analyze. If None, shows top performers.
    """
    # Calculate PER for all players
    df_with_per = CalculatePER(df)
    
    if player_name:
        # Analyze specific player
        player_stats = df_with_per[df_with_per['PLAYER_NAME'] == player_name]
        print(f"\nPER Statistics for {player_name}:")
        print(f"Average PER: {player_stats['PER'].mean():.2f}")
        print(f"Max PER: {player_stats['PER'].max():.2f}")
        print(f"Latest Rolling PER: {player_stats['Rolling_PER'].iloc[-1]:.2f}")
        
        # Show last 5 games
        print("\nLast 5 games:")
        recent_games = player_stats[['GAME_DATE', 'PER', 'Rolling_PER', 'MIN', 'PTS', 'REB', 'AST']].tail()
        print(recent_games)
        
    else:
        # Show top 10 performances by PER (minimum 15 minutes played)
        print("\nTop 10 Single-Game PER Performances (min. 15 minutes):")
        top_games = df_with_per[df_with_per['MIN'] >= 15].nlargest(10, 'PER')
        print(top_games[['PLAYER_NAME', 'GAME_DATE', 'PER', 'MIN', 'PTS', 'REB', 'AST']])
        
        # Show top 10 players by average PER (minimum 10 games played)
        print("\nTop 10 Players by Average PER (min. 10 games):")
        player_averages = df_with_per.groupby('PLAYER_NAME').agg({
            'PER': 'mean',
            'GAME_DATE': 'count'
        }).rename(columns={'GAME_DATE': 'Games_Played'})
        
        top_players = player_averages[player_averages['Games_Played'] >= 10].nlargest(10, 'PER')
        print(top_players)

# Example usage:
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # Analyze all players
    # analyze_player_per(df)
    
    # Analyze specific player
    # analyze_player_per(df, "LeBron James")
    
    print("To use this script:")
    print("1. Load your player data into a DataFrame")
    print("2. Call analyze_player_per(df) to see top performers")
    print("3. Call analyze_player_per(df, 'Player Name') to analyze a specific player") 