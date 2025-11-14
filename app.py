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
        if col in ['L-5', 'L-10', 'L-15'] or col.startswith('L-'):
            styled_df = styled_df.map(highlight_hit_rate, subset=[col])
    
    def format_percentage(x):
        """Format percentage columns (L-5, L-10, L-15) as percentages"""
        if pd.isna(x):
            return x
        if isinstance(x, (float, int)) and 0 <= x <= 1:
            return f'{x*100:.0f}%'
        return x
    
    def format_number(x):
        """Format regular numbers"""
        if pd.isna(x):
            return x
        return f'{x:g}' if pd.notna(x) else x
    
    format_dict = {}
    for col in df_copy.columns:
        # Format L-5, L-10, L-15 as percentages
        if col in ['L-5', 'L-10', 'L-15'] or (col.startswith('L-') and df_copy[col].dtype in ['float64', 'float32']):
            format_dict[col] = format_percentage
        # Format other numeric columns as regular numbers
        elif df_copy[col].dtype in ['float64', 'float32']:
            format_dict[col] = format_number
    
    if format_dict:
        styled_df = styled_df.format(format_dict)
    
    return styled_df

@st.cache_data  
def loadSinglePTSBookmakers(file_mtime):
    filepath = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV/singleBets.csv'
    df = pd.read_csv(filepath)
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadUnderdogPairsBookmakers(file_mtime):
    filepath = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogPairs.csv'
    df = pd.read_csv(filepath)
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadPrizepicksPairsBookmakers(file_mtime):
    filepath = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksPairs.csv'
    df = pd.read_csv(filepath)
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadTriosUnderdogBookmakers(file_mtime):
    filepath = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogTrios.csv'
    df = pd.read_csv(filepath)
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadTriosPrizepicksBookmakers(file_mtime):
    filepath = 'DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksTrios.csv'
    df = pd.read_csv(filepath)
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadOverRatesPrizePicks():
    over_rates_dir = 'DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS'
    over_rates_data = {}
    
    csv_files = sorted([f for f in os.listdir(over_rates_dir) if f.endswith('.csv')])
    
    # Use file list as part of cache key - cache invalidates when files change
    cache_key = tuple(csv_files)  # Make it hashable
    
    for file in csv_files:
        category = file.replace('.csv', '')
        filepath = os.path.join(over_rates_dir, file)
        try:
            over_rates_data[category] = pd.read_csv(filepath)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    
    return over_rates_data

@st.cache_data  
def loadOverRatesUnderdog():
    over_rates_dir = 'DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG'
    over_rates_data = {}
    
    csv_files = sorted([f for f in os.listdir(over_rates_dir) if f.endswith('.csv')])
    
    cache_key = tuple(csv_files)  
    
    for file in csv_files:
        category = file.replace('.csv', '')
        filepath = os.path.join(over_rates_dir, file)
        try:
            over_rates_data[category] = pd.read_csv(filepath)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
    
    return over_rates_data

# Get file modification times to use as cache keys
def get_file_mtime(filepath):
    return os.path.getmtime(filepath) if os.path.exists(filepath) else 0

# Streamlit UI
st.title('NBA Prop Predictions')

# Load data inside Streamlit execution flow so it recalculates on each rerun
singleBets = loadSinglePTSBookmakers(get_file_mtime('DATA/CSV_FILES/PROP_DATA/PROPS_EV/singleBets.csv'))
pairs = loadUnderdogPairsBookmakers(get_file_mtime('DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogPairs.csv'))
pairsPrizepicks = loadPrizepicksPairsBookmakers(get_file_mtime('DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksPairs.csv'))
triosUnderdog = loadTriosUnderdogBookmakers(get_file_mtime('DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogTrios.csv'))
triosPrizepicks = loadTriosPrizepicksBookmakers(get_file_mtime('DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksTrios.csv'))

# Header info with columns
st.write(f"**Predictions for {today}**")
st.write("**Creator:** Alex Gonzalez")
st.write("**Contact:** [Linkedin](https://www.linkedin.com/in/alex-gonzalez-data)")

# Description section
with st.expander("📋 What's Available", expanded=True):
    st.markdown("""
    ### **Betting Options**
    I created this app to look for the most profitable props(**points only**) across multiple platforms and to use as a tool to help me make decisions on my bets:
    
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
    - **0** = Not recommended (typically lower EV$ or higher risk)
    
    **Focus on bets with sigma flag = Low and high EV$ for the lowest risk opportunities.
    
    ### **Historical Hit Rates**
    The Historical Hit Rates section shows how often a player has hit their prop line historically:
    - **Last 5 Games**: Hit rate in the player's most recent 5 games
    - **Last 10 Games**: Hit rate in the player's most recent 10 games
    - **Last 15 Games**: Hit rate in the player's most recent 15 games
    
    This helps you understand a player's recent form and consistency for each prop type (points, rebounds, assists, etc.).
    
    ### **The Probability-Based Approach**
    
    Instead of saying *"Player X will score exactly 25 points"*, we calculate:
    - **Probability distribution**: How likely is it that Player X scores 20+ points? Given the uncertainty of the match and what we know about the player, we can create a probability distribution for the points scored.
    - **Model probability**: Based on historical data, matchups, and trends, what's the real probability?
    - **Odds probability**: What probability does the bookmaker's odds imply?
    - **Edge calculation**: If our probability > odds probability, we have positive expected value
    """)

st.info("⚠️ **Disclaimer**: Please take these predictions with a grain of salt. Sports betting involves significant uncertainty, and there are too many variables to accurately predict outcomes. **Bookmakers are sharp** - they use sophisticated models and set generally accurate odds that are difficult to beat consistently. Always gamble responsibly and within your means.")

# Add metric explanations in sidebar
with st.sidebar:
    st.header("Quick Reference")
    
    st.markdown("### EV Dollar Per 100$ Bet")
    st.markdown("""
    - **20$+**: Excellent EV
    - **10-19$**: Very good EV
    - **5-9$**: Good EV
    - **<5$**: Moderate EV
    - **Negative**: Avoid
    """)
    
    st.markdown("### Kelly Sizing")
    st.markdown("""
    - **Half**: Recommended
    - **Quarter**: Safest
    - **Full**: Aggressive only
    """)
    
    st.markdown("### Sigma Flags")
    st.markdown("""
    - **Low**: Lower risk
    - **Med**: Medium risk
    - **High**: Higher risk
    
    Focus on **Low/Med** flags.
    """)
    
    st.markdown("### Bet Selection")
    st.markdown("""
    1. High EV$ (top rows)
    2. Low sigma flags
    3. Check Over Rates
    4. Use Kelly Half/Quarter
    5. Max 5% per bet
    """)
    
    st.markdown("### Platforms")
    st.markdown("""
    **Single**: All US books
    
    **2-Leg/3-Leg**:
    - PrizePicks
    - Underdog
    """)

options = st.selectbox('Select a prop type', ['Single Bets', '2-Leg Bets', '3-Leg Bets', 'Historical Hit Rates - PrizePicks', 'Historical Hit Rates - Underdog'])

if options == 'Single Bets':
    st.subheader("Single Best Bets for Points")
    if singleBets.empty:
        st.warning("No data available for single bets. Please check if the data file exists.")
    else:
        styled_df = format_dataframe(singleBets.head(30)).reset_index(drop=True).style
        st.dataframe(styled_df, width='stretch')
    
elif options == '2-Leg Bets':
    st.subheader("Underdog Best Pairs for Points")
    if pairs.empty:
        st.warning("No bets available - Model found no profitable opportunities meeting criteria for Underdog pairs.")
    else:
        styled_pairs = format_dataframe(pairs.head(30)).reset_index(drop=True).style
        st.dataframe(styled_pairs, width='stretch')
    
    st.subheader("Prizepicks Best Pairs for Points")
    if pairsPrizepicks.empty:
        st.warning("No bets available - Model found no profitable opportunities meeting criteria for PrizePicks pairs.")
    else:
        styled_pairs_pp = format_dataframe(pairsPrizepicks.head(30)).reset_index(drop=True).style
        st.dataframe(styled_pairs_pp, width='stretch')
    
elif options == '3-Leg Bets':
    st.subheader("Underdog Best Trios for Points")
    if triosUnderdog.empty:
        st.warning("No bets available - Model found no profitable opportunities meeting criteria for Underdog trios.")
    else:
        styled_trios_ud = format_dataframe(triosUnderdog.head(30)).reset_index(drop=True).style
        st.dataframe(styled_trios_ud, width='stretch')
    
    st.subheader("Prizepicks Best Trios for Points")
    if triosPrizepicks.empty:
        st.warning("No bets available - Model found no profitable opportunities meeting criteria for PrizePicks trios.")
    else:
        styled_trios_pp = format_dataframe(triosPrizepicks.head(30)).reset_index(drop=True).style
        st.dataframe(styled_trios_pp, width='stretch')

elif options == 'Historical Hit Rates - PrizePicks':
    st.subheader("Over Rates for Player Props")
    over_rates = loadOverRatesPrizePicks()

    sections = [
        ('player_points', "Over Rates for Points"),
        ('player_rebounds', "Over Rates for Rebounds"),
        ('player_assists', "Over Rates for Assists"),
        ('player_blocks', "Over Rates for Blocks"),
        ('player_steals', "Over Rates for Steals"),
    ]

    any_rendered = False
    for key, title in sections:
        df = over_rates.get(key)
        if df is not None and not df.empty:
            st.subheader(title)
            
            if 'L-10' in df.columns:
                df_sorted = df.sort_values(by='L-10', ascending=False, na_position='last').reset_index(drop=True)
                st.dataframe(style_over_rates(df_sorted), width='stretch')
            else:
                # If column doesn't exist, show unsorted
                st.dataframe(style_over_rates(df), width='stretch')
            any_rendered = True

    if not any_rendered:
        st.info("No Over Rates available to display.")

elif options == 'Historical Hit Rates - Underdog':
    st.subheader("Over Rates for Player Props")
    over_rates = loadOverRatesUnderdog()
    sections = [
        ('player_points', "Over Rates for Points"),
        ('player_rebounds', "Over Rates for Rebounds"),
        ('player_assists', "Over Rates for Assists"),
        ('player_blocks', "Over Rates for Blocks"),
        ('player_steals', "Over Rates for Steals"),
    ]
    
    any_rendered = False
    for key, title in sections:
        df = over_rates.get(key)
        if df is not None and not df.empty:
            st.subheader(title)
          
            if 'L-10' in df.columns:
                df_sorted = df.sort_values(by='L-10', ascending=False, na_position='last').reset_index(drop=True)
                st.dataframe(style_over_rates(df_sorted), width='stretch')
            else:
                # If column doesn't exist, show unsorted
                st.dataframe(style_over_rates(df), width='stretch')
            any_rendered = True

    if not any_rendered:
        st.info("No Over Rates available to display.")