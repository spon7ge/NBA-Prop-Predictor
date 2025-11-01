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
    df = pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/singleBets.csv')
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadUnderdogPairsBookmakers():
    df = pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogPairs.csv')
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadPrizepicksPairsBookmakers():
    df = pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksPairs.csv')
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadTriosUnderdogBookmakers():
    df = pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogTrios.csv')
    return df if not df.empty else pd.DataFrame()

@st.cache_data  
def loadTriosPrizepicksBookmakers():
    df = pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksTrios.csv')
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

singleBets = loadSinglePTSBookmakers()
pairs = loadUnderdogPairsBookmakers()
pairsPrizepicks = loadPrizepicksPairsBookmakers()
triosUnderdog = loadTriosUnderdogBookmakers()
triosPrizepicks = loadTriosPrizepicksBookmakers()

# Streamlit UI
st.title('🏀 NBA Prop Predictions')

# Header info with columns
st.write(f"**Date:** {today}")
st.write("**Creator:** Alex Gonzalez")
st.write("**Contact:** alg21@stmarys-ca.edu")

# Update schedule info in an expandable info box
with st.expander("Update Schedule & Notes", expanded=False):
    st.info("""
    **Update Schedule:**
    - Updates posted daily at **12:00 PM PST**
    - Additional updates may occur before tip-off as new information becomes available
    
    **Important Note:**
    Odds are updated from bookmakers throughout the day as new information becomes available. 
    You may see different odds on the same bet depending on when you check the page.
    """)

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
    
    **Focus on bets with sigma flag = Low and high EV% for the lowest risk opportunities.
    
    ### **Historical Hit Rates**
    The Historical Hit Rates section shows how often a player has hit their prop line historically:
    - **Last 5 Games**: Hit rate in the player's most recent 5 games
    - **Last 10 Games**: Hit rate in the player's most recent 10 games
    - **Last 15 Games**: Hit rate in the player's most recent 15 games
    
    This helps you understand a player's recent form and consistency for each prop type (points, rebounds, assists, etc.).
    
    ### **The Probability-Based Approach**
    
    Instead of saying *"Player X will score exactly 25 points"*, we calculate:
    - **Probability distribution**: How likely is it that Player X scores 20+ points? 25+? 30+?
    - **Model probability**: Based on historical data, matchups, and trends, what's the real probability?
    - **Odds probability**: What probability does the bookmaker's odds imply?
    - **Edge calculation**: If our probability > odds probability, we have positive expected value
    
    ** What we're NOT doing:**
    - We're **NOT** trying to predict exactly how many points a player will score
    - We're **NOT** claiming to know the future outcome of any single game
    - Sports are inherently unpredictable - too many variables affect any individual game
    
    ** What we ARE doing:**
    - We're using **probability distributions** to estimate the likelihood of outcomes
    - We're identifying bets where the **probability of winning is higher than what the odds suggest**
    - We're finding **expected value** - bets that are profitable over many repetitions
    - We're applying mathematical edge-finding to make profitable decisions **long-term**
    
    """)

# Disclaimer
st.info("⚠️ **Disclaimer**: Please take these predictions/probabilities with a grain of salt. Sports betting involves significant uncertainty, and there are too many variables to accurately predict outcomes. **Bookmakers are sharp** - they use sophisticated models and set generally accurate odds that are difficult to beat consistently. Always gamble responsibly and within your means.")

# Add expandable help section with tooltips
with st.expander("Betting Metrics Explained", expanded=False):
    st.markdown("""
    ### **EV% (Expected Value Percentage)**
    The average profit percentage per bet over many repetitions. Calculated by comparing our model's win probability vs. the bookmaker's implied probability from odds.
    
    - **Positive EV%** = Profitable over time (higher is better)
    - **Example**: 25% EV means +25 profit per 100 dollars wagered on average
    - Higher EV% bets appear at the top of each table
    
    ### **Kelly Criterion**
    Mathematical formula for optimal bet sizing based on your edge. Maximizes long-term growth while protecting your bankroll.
    
    - **Kelly Full**: Aggressive (max recommended) - for experienced bettors only
    - **Kelly Half**: Conservative (recommended) - bet 50% of Kelly suggestion
    - **Kelly Quarter**: Very safe - bet 25% of Kelly suggestion
    
    Most bettors should use Kelly Half or Quarter for safety.
    
    ### **How the Model Works**
    Uses NGBoost (probabilistic ML) trained on thousands of games to estimate **probability distributions**, not exact scores. Considers:
    
    - Player performance (rolling averages, trends, volatility)
    - Game context (home/away, rest days, back-to-back)
    - Matchups (opponent defense, historical performance, team pace)
    - Team context (ratings, star availability, recent form)
    
    Compares the model's probability distribution to betting odds to identify value opportunities.
    
    ### **Uncertainty Metrics**
    Indicators of prediction confidence:
    
    - **CONFIDENCE INTERVAL**: Range where outcome likely falls (e.g., 95% confidence)
    - **INTERVAL WIDTH**: Size of that range (smaller = more certain)
    - **SIGMA**: Standard deviation of the distribution (higher = more volatile)
    - **SIGMA FLAG**: Alert when volatility is elevated (Low/Med/High)
    
    Focus on bets with **Low sigma flags** for lower risk opportunities.
    """)

# Add metric explanations in sidebar
with st.sidebar:
    st.header("Quick Reference")
    
    st.markdown("### EV% Guide")
    st.markdown("""
    - **20%+**: Excellent
    - **10-19%**: Very good
    - **5-9%**: Good
    - **<5%**: Moderate
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
    
    Focus on **Low** flags.
    """)
    
    st.markdown("### Bet Selection")
    st.markdown("""
    1. High EV% (top rows)
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
            # Sort by HIT RATE % LAST 10 in descending order (highest first)
            if 'HIT RATE % LAST 10' in df.columns:
                df_sorted = df.sort_values(by='HIT RATE % LAST 10', ascending=False, na_position='last').reset_index(drop=True)
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
            # Sort by HIT RATE % LAST 10 in descending order (highest first)
            if 'HIT RATE % LAST 10' in df.columns:
                df_sorted = df.sort_values(by='HIT RATE % LAST 10', ascending=False, na_position='last').reset_index(drop=True)
                st.dataframe(style_over_rates(df_sorted), width='stretch')
            else:
                # If column doesn't exist, show unsorted
                st.dataframe(style_over_rates(df), width='stretch')
            any_rendered = True

    if not any_rendered:
        st.info("No Over Rates available to display.")