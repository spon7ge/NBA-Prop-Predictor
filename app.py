import pandas as pd
import streamlit as st
import joblib
from PROPS_EV.calculateEVS import *
from MODELS.pipeline import *
from datetime import datetime

# today = datetime.today().strftime('%Y%m%d')
today = '20251019'

def loadModel():
    return joblib.load('MODELS/Models/xgbPTSModelNoOppStats26.pkl')

def loadFeatures():
    return joblib.load('MODELS/Models/topPTSfeaturesNoOppStats26.pkl')

def loadData():
    return pd.read_csv('DATA/CSV_FILES/TRAIN_DATA/PTS_TRAIN_25.csv')

def loadBookmakers():
    return pd.read_csv(f'DATA/CSV_FILES/PROP_DATA/PLAYER_LINES/NBA_DFS_{today}.csv')


def getPredictions(data, bookmakers, _model, _features):
    # singles = single_bet(data, bookmakers, _model, _features, edge_threshold=4.5, stake=10, simulations=10000, std_window=10, min_std=2.0, max_std=10.0, stat_col='PTS')
    pairs = prizepickspairsEV(data, bookmakers, _model, _features, edge_threshold=4.5, stake=10, simulations=10000, std_window=10, min_std=2.0, max_std=10.0, stat_col='PTS')
    trios = prizepicks3LegEV(data, bookmakers, _model, _features, edge_threshold=4.5, stake=10, simulations=10000, std_window=10, min_std=2.0, max_std=10.0, stat_col='PTS')
    return pairs, trios

model = loadModel()
data = loadData()
features = loadFeatures()
bookmakers = loadBookmakers()
pairs, trios = getPredictions(data, bookmakers, model, features)

st.title('NBA Prop Predictions')
st.write(f'Today is {today}')
st.write(f'Predictions for {today}')

st.subheader("2 Legs")
st.dataframe(pairs[pairs["RECOMMENDATION"] == 1].sort_values("EV%", ascending=False))

st.subheader("3 Legs")
st.dataframe(trios[trios["RECOMMENDATION"] == 1].sort_values("EV%", ascending=False))

