"""Shared leakage-safe helpers used by NBA / WNBA feature modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-window fatigue proxies from prior games only (excludes current game)."""
    df = df.copy()
    games_7 = np.zeros(len(df), dtype=int)
    games_14 = np.zeros(len(df), dtype=int)
    min_sum_7 = np.zeros(len(df), dtype=float)

    for idx in df.groupby("PLAYER_ID").groups.values():
        pos = df.index.get_indexer(idx)
        order = np.argsort(df.loc[idx, "GAME_DATE"].to_numpy())
        pos = pos[order]
        dates = df.loc[idx, "GAME_DATE"].to_numpy(dtype="datetime64[ns]")[order]
        mins = df.loc[idx, "MIN"].to_numpy(dtype=float)[order]

        for i in range(1, len(pos)):
            d = dates[i]
            prior_dates = dates[:i]
            prior_mins = mins[:i]
            mask_7 = prior_dates >= (d - np.timedelta64(7, "D"))
            mask_14 = prior_dates >= (d - np.timedelta64(14, "D"))
            games_7[pos[i]] = mask_7.sum()
            games_14[pos[i]] = mask_14.sum()
            min_sum_7[pos[i]] = prior_mins[mask_7].sum()

    df["GAMES_PLAYED_LAST_7_DAYS"] = games_7
    df["GAMES_PLAYED_LAST_14_DAYS"] = games_14
    df["MIN_SUM_LAST_7_DAYS"] = min_sum_7.round(1)
    return df


def days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Days since prior game for each player (first game defaults to 3)."""
    df = df.copy()
    df["DAYS_REST"] = (
        df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.fillna(3)
    )
    return df
