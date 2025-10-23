import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from datetime import datetime
import numpy as np

today = datetime.today().strftime('%m-%d-%Y')  

@st.cache_data  
def loadSinglePTSBookmakers():
    return pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/singleBets_20251021.csv')

@st.cache_data  
def loadUnderdogPairsBookmakers():
    return pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdog_20251021.csv')

@st.cache_data  
def loadPrizepicksPairsBookmakers():
    return pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicks_20251021.csv')

@st.cache_data  
def loadTriosUnderdogBookmakers():
    return pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogTrios_20251021.csv')

@st.cache_data  
def loadTriosPrizepicksBookmakers():
    return pd.read_csv('DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksTrios_20251021.csv')

# Function to apply row-level color coding based on EV% and recommendation
def highlight_rows(row):
    styles = [''] * len(row)
    
    # Check if this is a recommended bet (RECOMMENDATION == 1)
    is_recommended = row.get('RECOMMENDATION', 0) == 1
    
    # Get EV% value
    ev_value = row.get('EV%', 0)
    try:
        ev = float(ev_value) if pd.notna(ev_value) else 0
    except:
        ev = 0
    
    # Determine row color based on EV% and recommendation
    if is_recommended and ev >= 50:  # High EV + Recommended - Dark Green
        color = 'background-color: #d4edda; color: #155724; font-weight: bold'
    elif is_recommended and ev >= 20:  # Medium EV + Recommended - Light Green
        color = 'background-color: #c3e6cb; color: #155724'
    elif is_recommended and ev >= 0:  # Low positive EV + Recommended - Very Light Green
        color = 'background-color: #e8f5e8; color: #155724'
    elif ev >= 50:  # High EV but not recommended - Yellow
        color = 'background-color: #fff3cd; color: #856404; font-weight: bold'
    elif ev >= 20:  # Medium EV but not recommended - Light Yellow
        color = 'background-color: #fef9e7; color: #856404'
    elif ev < 0:  # Negative EV - Red
        color = 'background-color: #f8d7da; color: #721c24'
    else:  # Default
        color = ''
    
    # Apply the same color to all cells in the row
    return [color] * len(row)

singleBets = loadSinglePTSBookmakers()
pairs = loadUnderdogPairsBookmakers()
pairsPrizepicks = loadPrizepicksPairsBookmakers()
triosUnderdog = loadTriosUnderdogBookmakers()
triosPrizepicks = loadTriosPrizepicksBookmakers()

# Streamlit UI
st.title('NBA Prop Predictions')
st.write(f'Predictions for {today}')

# Add expandable help section with tooltips
with st.expander("Betting Metrics Explained", expanded=False):
    st.markdown("""
    ### **EV% (Expected Value Percentage)**
    - **What it is**: The percentage profit you can expect to make on average per bet
    - **How to read**: 
      - **Green (≥50%)**: Excellent value - high probability of profit
      - **Yellow (20-49%)**: Good value - moderate profit potential  
      - **Light Green (0-19%)**: Positive value - small profit potential
      - **Red (<0%)**: Negative value - avoid these bets
    - **Example**: EV% of 25% means you expect to make 25 dollars in profit for every 100 dollars wagered
    
    ### **Kelly Criterion**
    - **What it is**: A formula that tells you what percentage of your bankroll to bet
    - **KELLY FULL**: Maximum recommended bet size (aggressive)
    - **KELLY HALF**: Conservative bet size (recommended for beginners)
    - **KELLY QUARTER**: Very conservative bet size (safest option)
    - **How to use**: 
      - If Kelly suggests 10%, bet 10% of your bankroll
      - Never bet more than Kelly suggests
      - Start with Kelly Half or Quarter for safety
    
    ### **Color Coding Guide**
    - **🟢 Dark Green**: Recommended + High EV (≥50%) - Best bets
    - **🟢 Light Green**: Recommended + Medium/Low EV - Good bets
    - **🟡 Yellow**: High EV but not recommended - Proceed with caution
    - **🔴 Red**: Negative EV - Avoid these bets
    """)

# Add metric explanations in sidebar
with st.sidebar:
    st.header("Quick Reference")
    
    st.markdown("### EV% Guide")
    st.markdown("""
    - **≥50%**: 🟢 Excellent
    - **20-49%**: 🟡 Good  
    - **0-19%**: 🟢 Positive
    - **<0%**: 🔴 Avoid
    """)
    
    st.markdown("### Kelly Criterion")
    st.markdown("""
    - **Full**: Max bet size
    - **Half**: Conservative
    - **Quarter**: Safest
    """)
    
    st.markdown("### Pro Tips")
    st.markdown("""
    1. Focus on green highlighted bets
    2. Start with Kelly Half sizing
    3. Never bet more than 5% of bankroll
    4. Track your results
    """)

options = st.selectbox('Select a prop type', ['Single Bets', '2-Leg Bets', '3-Leg Bets'])

if options == 'Single Bets':
    st.subheader("Single Best Bets for Points")
    # Apply row-level color coding to the dataframe
    styled_df = singleBets.head(30).reset_index().style.apply(highlight_rows, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
elif options == '2-Leg Bets':
    st.subheader("Underdog Best Pairs for Points")
    styled_pairs = pairs.head(30).reset_index().style.apply(highlight_rows, axis=1)
    st.dataframe(styled_pairs, use_container_width=True)
    
    st.subheader("Prizepicks Best Pairs for Points")
    styled_pairs_pp = pairsPrizepicks.head(30).reset_index().style.apply(highlight_rows, axis=1)
    st.dataframe(styled_pairs_pp, use_container_width=True)
    
elif options == '3-Leg Bets':
    st.subheader("Underdog Best Trios for Points")
    styled_trios_ud = triosUnderdog.head(30).reset_index().style.apply(highlight_rows, axis=1)
    st.dataframe(styled_trios_ud, use_container_width=True)
    
    st.subheader("Prizepicks Best Trios for Points")
    styled_trios_pp = triosPrizepicks.head(30).reset_index().style.apply(highlight_rows, axis=1)
    st.dataframe(styled_trios_pp, use_container_width=True)