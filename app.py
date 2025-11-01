import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
import os
from zoneinfo import ZoneInfo

# Use PST timezone for date calculation
pst = ZoneInfo("America/Los_Angeles")
now_pst = datetime.now(pst)
today = now_pst.strftime('%m-%d-%Y')  
bet_date = now_pst.strftime('%Y%m%d')

def find_latest_file(base_path, filename_prefix):
    """Find the most recent file matching the prefix, trying today's date first."""
    import glob
    
    # First try today's date
    today_file = f'{base_path}/{filename_prefix}_{bet_date}.csv'
    if os.path.exists(today_file):
        return today_file
    
    # If not found, search for the most recent file
    pattern = f'{base_path}/{filename_prefix}_*.csv'
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by filename (which includes date) and return the most recent
    files.sort(reverse=True)
    return files[0]

def format_dataframe(df):
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if df_formatted[col].dtype in ['float64', 'float32']:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f'{x:g}' if pd.notna(x) else x
            )
    return df_formatted

def style_over_rates(df):
    """Apply conditional formatting to over rates dataframes based on hit rate percentages"""
    df_copy = df.copy()
    
    def highlight_hit_rate(val):
        if pd.isna(val):
            return ''
    
        if val > 0.8: 
            return 'background-color: #90EE90; color: black; font-weight: bold'  # Light green
        elif val >= 0.5:  
            return 'background-color: #FFF59D; color: black; font-weight: bold'  # Lighter yellow
        else:  
            return 'background-color: #FF8C69; color: black; font-weight: bold'  # Salmon/coral
    
    styled_df = df_copy.style
    for col in df_copy.columns:
        if 'HIT RATE' in col.upper():
            styled_df = styled_df.map(highlight_hit_rate, subset=[col])
    
    def format_number(x):
        return f'{x:g}' if pd.notna(x) else x
    
    format_dict = {}
    for col in df_copy.columns:
        if df_copy[col].dtype in ['float64', 'float32']:
            format_dict[col] = format_number
    
    # Apply formatting
    if format_dict:
        styled_df = styled_df.format(format_dict)
    
    return styled_df

@st.cache_data  
def loadSinglePTSBookmakers():
    base_path = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV'
    file_path = find_latest_file(base_path, 'singleBets')
    if file_path is None:
        st.error(f"No data file found for single bets. Expected: singleBets_{bet_date}.csv")
        return pd.DataFrame()  # Return empty dataframe
    # Show info if using a different date than today
    if bet_date not in file_path:
        file_date = os.path.basename(file_path).replace('singleBets_', '').replace('.csv', '')
        st.info(f"⚠️ Using data from {file_date} (today's file not available)")
    return pd.read_csv(file_path)

@st.cache_data  
def loadUnderdogPairsBookmakers():
    base_path = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV'
    file_path = find_latest_file(base_path, 'underdogPairs')
    if file_path is None:
        st.error(f"No data file found for Underdog pairs. Expected: underdogPairs_{bet_date}.csv")
        return pd.DataFrame()
    return pd.read_csv(file_path)

@st.cache_data  
def loadPrizepicksPairsBookmakers():
    base_path = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV'
    file_path = find_latest_file(base_path, 'prizepicksPairs')
    if file_path is None:
        st.error(f"No data file found for PrizePicks pairs. Expected: prizepicksPairs_{bet_date}.csv")
        return pd.DataFrame()
    return pd.read_csv(file_path)

@st.cache_data  
def loadTriosUnderdogBookmakers():
    base_path = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV'
    file_path = find_latest_file(base_path, 'underdogTrios')
    if file_path is None:
        st.error(f"No data file found for Underdog trios. Expected: underdogTrios_{bet_date}.csv")
        return pd.DataFrame()
    return pd.read_csv(file_path)

@st.cache_data  
def loadTriosPrizepicksBookmakers():
    base_path = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV'
    file_path = find_latest_file(base_path, 'prizepicksTrios')
    if file_path is None:
        st.error(f"No data file found for PrizePicks trios. Expected: prizepicksTrios_{bet_date}.csv")
        return pd.DataFrame()
    return pd.read_csv(file_path)

@st.cache_data  
def loadOverRates():

    over_rates_dir = 'DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS'
    over_rates_data = {}
            
    csv_files = [f for f in os.listdir(over_rates_dir) if f.endswith('.csv')]
    for file in csv_files:
        category = file.replace('.csv', '')
        filepath = os.path.join(over_rates_dir, file)
        try:
            over_rates_data[category] = pd.read_csv(filepath)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    
    return over_rates_data


singleBets = loadSinglePTSBookmakers()
pairs = loadUnderdogPairsBookmakers()
pairsPrizepicks = loadPrizepicksPairsBookmakers()
triosUnderdog = loadTriosUnderdogBookmakers()
triosPrizepicks = loadTriosPrizepicksBookmakers()

# Streamlit UI
st.title('NBA Prop Predictions')
st.write(f'Predictions for {today}')

# Description section
with st.expander("📋 What's Available", expanded=True):
    st.markdown("""
    ### **Betting Options**
    This app provides prop betting opportunities for **points only** across multiple platforms:
    
    - **Single Bets**: Individual player prop bets with the best expected value
    - **2-Leg Bets**: Two-player combinations available on:
      - PrizePicks
      - Underdog
    - **3-Leg Bets**: Three-player combinations available on:
      - PrizePicks
      - Underdog
    
    ### **Recommendation Column**
    Each betting table includes a **RECOMMENDATION** column that identifies the best bets:
    - **1** = **Recommended Bet**: The model has high confidence this bet offers strong value
    - **0** = Not recommended (typically lower EV% or higher risk)
    
    **Focus on bets with RECOMMENDATION = 1** for the highest quality opportunities. These bets have been filtered through multiple criteria including EV%, probability estimates, and risk assessment.
    
    ### **Over Rates**
    The Over Rates section shows how often a player has hit their prop line historically:
    - **Last 5 Games**: Hit rate in the player's most recent 5 games
    - **Last 10 Games**: Hit rate in the player's most recent 10 games
    - **Last 15 Games**: Hit rate in the player's most recent 15 games
    
    This helps you understand a player's recent form and consistency for each prop type (points, rebounds, assists, etc.).
    
    The color coding helps you quickly identify performance:
    - 🟢 **Green** (>80%): Excellent recent performance
    - 🟡 **Yellow** (50-79%): Good recent performance
    - 🔴 **Salmon** (<50%): Below average recent performance
    """)

# Disclaimer
st.info("⚠️ **Disclaimer**: Please take these predictions with a grain of salt. Sports betting involves significant uncertainty, and anything can happen in any given game. Past performance does not guarantee future results. Always gamble responsibly and within your means.")

# Add expandable help section with tooltips
with st.expander("Betting Metrics Explained", expanded=False):
    st.markdown("""
    ### **EV% (Expected Value Percentage)**
    **What it is**: The percentage profit you can expect to make on average per bet. It's calculated by comparing the predicted probability of winning (from your model) with the implied probability from the betting odds.
    
    **How it works**:
    - If your model says a bet has a 60% chance to win, but the odds imply only a 50% chance, you have positive expected value
    - EV% tells you the average profit percentage per bet if you made this bet repeatedly
    - **Example**: EV% of 25% means you expect to make 25 dollars in profit for every 100 dollars wagered over the long run
    
    **Interpretation**:
    - Positive EV% = Profitable bet (the higher, the better)
    - Negative EV% = Unprofitable bet (avoid these)
    - Higher EV% bets are ranked at the top of each table
    
    ### **Kelly Criterion**
    **What it is**: A mathematical formula that tells you what percentage of your bankroll to bet based on your edge (advantage) and the odds. It maximizes long-term growth while protecting your bankroll.
    
    **Why it matters**: Betting too much risks going broke; betting too little leaves money on the table. Kelly finds the optimal balance.
    
    **Betting Sizes**:
    - **KELLY FULL**: Maximum recommended bet size (aggressive) - Use only if you're very confident and experienced
    - **KELLY HALF**: Conservative bet size (recommended for beginners) - Bet half of what Kelly suggests
    - **KELLY QUARTER**: Very conservative bet size (safest option) - Bet a quarter of what Kelly suggests
    
    **How to use**: 
    - If Kelly Full suggests 10%, bet 10% of your bankroll
    - If Kelly Half suggests 10%, bet 5% of your bankroll  
    - If Kelly Quarter suggests 10%, bet 2.5% of your bankroll
    - Never bet more than Kelly Full suggests
    - Most bettors should start with Kelly Half or Quarter for safety
    
    ### **What the Model Uses to Make Predictions**
    The model uses an NGBoost machine learning algorithm trained on thousands of NBA games. Here's what data goes into each prediction:
    
    **Player Performance Features**:
    - Rolling averages of points, shooting percentages, usage rate, and minutes over recent games (3, 5, 7, 10+ game windows)
    - Historical performance patterns and trends
    - Recent form indicators (hot/cold streaks, consistency metrics)
    - Performance volatility (how consistent the player has been)
    
    **Game Context**:
    - **Home vs Away**: Players often perform differently at home vs on the road
    - **Rest Days**: How many days of rest the player and team have had (affects performance)
    - **Back-to-Back Games**: Whether the team is playing consecutive nights
    - **Starting Lineup**: Whether the player is projected to start
    
    **Matchup Features**:
    - **Opponent Defense**: How well the opponent defends against the player's position
    - **Historical Matchups**: How the player has performed against this specific opponent before
    - **Team Pace**: Whether it's a fast-paced or slow-paced game (more opportunities vs fewer)
    
    **Team Context**:
    - Team offensive and defensive ratings
    - Whether star teammates are playing (affects usage and opportunities)
    - Team recent form and momentum
    
    **Advanced Metrics**:
    - True shooting percentage, effective field goal percentage
    - Usage rate and touches per game
    - Efficiency ratings and net rating contributions
    
    The model learns from all these factors to predict how many points a player will score, then compares that prediction to the betting line to find value bets.
    
    ### **Uncertainty Metrics**
    These columns help you gauge how tight or uncertain a projection is:

    - **CONFIDENCE INTERVAL**: The projected range for a player's outcome (e.g., points) within a chosen confidence level (commonly 95%). Shows the lower and upper bounds where the result is expected to fall.
    - **INTERVAL WIDTH**: The size of that range (Upper − Lower). Smaller width = higher certainty; larger width = more uncertainty.
    - **SIGMA**: The model's estimated standard deviation of the projection distribution (in points). Higher sigma indicates more volatile outcomes.
    - **SIGMA FLAG**: A quick alert when volatility is elevated (e.g., when sigma exceeds a threshold or data quality is limited). Use flagged rows with extra caution.
    """)

# Add metric explanations in sidebar
with st.sidebar:
    st.header("Quick Reference")
    
    st.markdown("### EV% Guide")
    st.markdown("""
    - **20%+**: Excellent bet
    - **10-19%**: Very good bet
    - **5-9%**: Good bet
    - **<5%**: Moderate bet
    - **Negative**: Avoid
    """)
    
    st.markdown("### Kelly Sizing")
    st.markdown("""
    - **Full**: Max aggressive
    - **Half**: Recommended
    - **Quarter**: Safest
    """)
    
    st.markdown("### Over Rates Colors")
    st.markdown("""
    - 🟢 **Green** (>80%): Excellent
    - 🟡 **Yellow** (50-79%): Good  
    - 🔴 **Salmon** (<50%): Poor
    """)
    
    st.markdown("### Recommendation Column")
    st.markdown("""
    - **1** = Recommended bet
    - **0** = Not recommended
    
    **Filter by RECOMMENDATION = 1** for best bets
    """)
    
    st.markdown("### Bet Selection Tips")
    st.markdown("""
    1. **Filter REC = 1**: Best quality bets
    2. **Prioritize High EV%**: Top = best value
    3. **Check Over Rates**: Green = confident
    4. **Use Kelly Half**: Safe sizing
    5. **Bankroll Rule**: Max 5% per bet
    """)
    
    st.markdown("### Available Platforms")
    st.markdown("""
    **Single Bets**: All US bookmakers
    
    **2-Leg & 3-Leg**:
    - PrizePicks
    - Underdog  
    """)

options = st.selectbox('Select a prop type', ['Single Bets', '2-Leg Bets', '3-Leg Bets', 'Over Rates PrizePicks'])

if options == 'Single Bets':
    st.subheader("Single Best Bets for Points")
    if singleBets.empty:
        st.warning("No data available for single bets. Please check if the data file exists.")
    else:
        styled_df = format_dataframe(singleBets.head(30)).reset_index(drop=True).style
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
elif options == '2-Leg Bets':
    st.subheader("Underdog Best Pairs for Points")
    if pairs.empty:
        st.warning("No data available for Underdog pairs. Please check if the data file exists.")
    else:
        styled_pairs = format_dataframe(pairs.head(30)).reset_index(drop=True).style
        st.dataframe(styled_pairs, use_container_width=True, hide_index=True)
    
    st.subheader("Prizepicks Best Pairs for Points")
    if pairsPrizepicks.empty:
        st.warning("No data available for PrizePicks pairs. Please check if the data file exists.")
    else:
        styled_pairs_pp = format_dataframe(pairsPrizepicks.head(30)).reset_index(drop=True).style
        st.dataframe(styled_pairs_pp, use_container_width=True, hide_index=True)
    
elif options == '3-Leg Bets':
    st.subheader("Underdog Best Trios for Points")
    if triosUnderdog.empty:
        st.warning("No data available for Underdog trios. Please check if the data file exists.")
    else:
        styled_trios_ud = format_dataframe(triosUnderdog.head(30)).reset_index(drop=True).style
        st.dataframe(styled_trios_ud, use_container_width=True, hide_index=True)
    
    st.subheader("Prizepicks Best Trios for Points")
    if triosPrizepicks.empty:
        st.warning("No data available for PrizePicks trios. Please check if the data file exists.")
    else:
        styled_trios_pp = format_dataframe(triosPrizepicks.head(30)).reset_index(drop=True).style
        st.dataframe(styled_trios_pp, use_container_width=True, hide_index=True)

elif options == 'Over Rates PrizePicks':
    st.subheader("Over Rates for Player Props")
    over_rates = loadOverRates()

    sections = [
        ('player_points', "Over Rates for Points"),
        ('player_rebounds', "Over Rates for Rebounds"),
        ('player_assists', "Over Rates for Assists"),
        ('player_blocks', "Over Rates for Blocks"),
        ('player_steals', "Over Rates for Steals"),
        ('player_turnovers', "Over Rates for Turnovers"),
        ('player_field_goals', "Over Rates for Field Goals Made"),
        ('player_frees_made', "Over Rates for Free Throws Made"),
        ('player_threes', "Over Rates for Three Pointers Made"),
        ('player_points_rebounds', "Over Rates for Points + Rebounds"),
        ('player_points_assists', "Over Rates for Points + Assists"),
        ('player_rebounds_assists', "Over Rates for Rebounds + Assists"),
        ('player_points_rebounds_assists', "Over Rates for Points + Rebounds + Assists"),
    ]

    any_rendered = False
    for key, title in sections:
        df = over_rates.get(key)
        if df is not None and not df.empty:
            st.subheader(title)
            st.dataframe(style_over_rates(df), use_container_width=True, hide_index=True)
            any_rendered = True

    if not any_rendered:
        st.info("No Over Rates available to display.")