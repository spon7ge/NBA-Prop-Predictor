"""Train / holdout and date-safe walk-forward split helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd

from models.shared.baselines import ensure_naive_cols


def season_holdout_split(
    df: pd.DataFrame,
    *,
    holdout_season: str,
    season_col: str = "season_year",
    date_col: str = "game_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into train pool vs locked holdout season (sorted by date)."""
    train = (
        df[df[season_col] != holdout_season]
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    holdout = (
        df[df[season_col] == holdout_season]
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    return train, holdout


def date_walk_forward_folds(
    df: pd.DataFrame,
    *,
    date_col: str = "game_date",
    train_frac: float = 0.50,
    step_frac: float = 0.10,
    n_folds: int = 4,
) -> Iterator[dict[str, Any]]:
    """Yield date-based walk-forward folds (no intra-game leakage).

    Every row sharing a ``game_date`` lands entirely in train or entirely in val.
    """
    if not df[date_col].is_monotonic_increasing:
        raise ValueError(f"{date_col} must be sorted ascending before walk-forward")

    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)
    train_window = round(n_dates * train_frac)
    step_size = round(n_dates * step_frac)

    for fold in range(n_folds):
        d_train_start = fold * step_size
        d_train_end = d_train_start + train_window
        d_val_end = d_train_end + step_size
        if d_val_end > n_dates:
            break

        train_dates = unique_dates[d_train_start:d_train_end]
        val_dates = unique_dates[d_train_end:d_val_end]
        train_mask = df[date_col].isin(train_dates)
        val_mask = df[date_col].isin(val_dates)

        yield {
            "fold": fold + 1,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "train_dates": train_dates,
            "val_dates": val_dates,
            "label": (
                f"WF fold {fold + 1}  "
                f"({pd.Timestamp(train_dates[-1]).date()} → "
                f"{pd.Timestamp(val_dates[-1]).date()})"
            ),
        }


def prepare_splits(
    df: pd.DataFrame,
    *,
    holdout_season: str,
    features: Sequence[str],
    target_col: str,
    naive_primary: str,
    naive_secondary: str,
    id_cols: Sequence[str],
    role_col: str = "starting",
    extra_keep: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build train pool / holdout frames and feature matrices for one prop."""
    features = list(features)
    naive_cols = [naive_primary, naive_secondary]
    df = ensure_naive_cols(
        df,
        target_col=target_col,
        primary_col=naive_primary,
        secondary_col=naive_secondary,
    )

    train_pool, holdout = season_holdout_split(df, holdout_season=holdout_season)

    keep_cols = list(dict.fromkeys(
        list(features)
        + list(id_cols)
        + [target_col, role_col]
        + list(extra_keep or [])
        + naive_cols
    ))
    missing_naive = [c for c in naive_cols if c not in df.columns]
    if missing_naive:
        raise KeyError(f"Naive baseline cols missing after ensure: {missing_naive}")

    train_df = (
        train_pool[keep_cols]
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    holdout_df = (
        holdout[keep_cols]
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    X = train_df[features].copy()
    y = train_df[target_col]

    print(f"Train pool (excl. {holdout_season}) : {len(train_df):,} rows")
    print(f"Holdout   ({holdout_season})        : {len(holdout_df):,} rows")
    print(
        f"Date range (train)   : "
        f"{train_df['game_date'].min().date()} → {train_df['game_date'].max().date()}"
    )
    print(
        f"Date range (holdout) : "
        f"{holdout_df['game_date'].min().date()} → "
        f"{holdout_df['game_date'].max().date()}"
    )
    return {
        "train_df": train_df,
        "holdout_df": holdout_df,
        # aliases used by older notebooks
        "ppm_df": train_df,
        "ppm_holdout": holdout_df,
        "X": X,
        "y": y,
        "features": features,
    }
