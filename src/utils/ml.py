"""ML layer: feature IO, quantile training, and prediction helpers."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.apm_features import APM_FEATURES
from src.features.min_features import MIN_FEATURES
from src.features.ppm_features import PPM_FEATURES
from src.features.rpm_features import RPM_FEATURES

ML_PROP_TABLES = {
    "min": "features_min",
    "ppm": "features_ppm",
    "rpm": "features_rpm",
    "apm": "features_apm",
}

ML_PROP_FEATURES = {
    "min": MIN_FEATURES,
    "ppm": PPM_FEATURES,
    "rpm": RPM_FEATURES,
    "apm": APM_FEATURES,
}

ML_PROP_TARGETS = {
    "min": "MIN",
    "ppm": "PTS_PER_MIN",
    "rpm": "REB_PER_MIN",
    "apm": "AST_PER_MIN",
}

ML_QUANTILES = (0.10, 0.50, 0.90)

ML_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:quantileerror",
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 50,
}

_GOLD_KEYS = ("GAME_ID", "PLAYER_ID")
_META_COLS = ("GAME_DATE", "PLAYER_NAME", "STARTING", "MATCHUP", "SEASON_YEAR")
_DEFAULT_MODELS_DIR = Path("src/models/saved_models")

_TRAINING_FILTERS = {
    "min": lambda df: (df["MIN"] >= 5) & (df["MIN"] <= 48),
    "ppm": lambda df: (df["MIN"].clip(upper=48) >= 10) | (df["STARTING"] == 1),
    "rpm": lambda df: (df["MIN"].clip(upper=48) >= 10) | (df["STARTING"] == 1),
    "apm": lambda df: (df["MIN"].clip(upper=48) >= 15) | (df["STARTING"] == 1),
}


def _canonical_col(name: str) -> str:
    """Normalize DB / Python feature names for matching."""
    s = str(name).strip().lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_canonical_col(c) for c in out.columns]
    return out


def align_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return a feature matrix with columns ordered exactly as ``feature_names``."""
    canon = _normalize_columns(df)
    lookup = {c: c for c in canon.columns}
    out = pd.DataFrame(index=canon.index)
    for feat in feature_names:
        key = _canonical_col(feat)
        if key in lookup:
            out[feat] = canon[key]
        else:
            out[feat] = np.nan
    return out


def read_ml_features(
    prop: str,
    *,
    season_year: str | None = None,
    season_type: str = "Regular Season",
    game_date: str | None = None,
) -> pd.DataFrame:
    """Load one prop's ML feature table from Supabase."""
    from src.utils.db import read_df

    prop = prop.lower()
    if prop not in ML_PROP_TABLES:
        raise ValueError(f"Unknown prop {prop!r}; expected one of {sorted(ML_PROP_TABLES)}")

    clauses = ["season_type = %(season_type)s"]
    params: dict[str, str] = {"season_type": season_type}
    if season_year is not None:
        clauses.append("season_year = %(season_year)s")
        params["season_year"] = season_year
    if game_date is not None:
        clauses.append("game_date = %(game_date)s")
        params["game_date"] = game_date

    df = read_df(
        ML_PROP_TABLES[prop],
        schema="ml",
        where=" and ".join(clauses),
        params=params,
    )
    if df.empty:
        feature_cols = ML_PROP_FEATURES[prop]
        target = ML_PROP_TARGETS[prop]
        return pd.DataFrame(columns=[*_GOLD_KEYS, *_META_COLS, *feature_cols, target])

    out = _normalize_columns(df)
    rename = {
        "game_id": "GAME_ID",
        "player_id": "PLAYER_ID",
        "player_name": "PLAYER_NAME",
        "game_date": "GAME_DATE",
        "season_year": "SEASON_YEAR",
        "season_type": "SEASON_TYPE",
        "starting": "STARTING",
        "matchup": "MATCHUP",
        "min": "MIN",
        "pts_per_min": "PTS_PER_MIN",
        "reb_per_min": "REB_PER_MIN",
        "ast_per_min": "AST_PER_MIN",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def ml_training_columns(prop: str) -> list[str]:
    """Feature + target + metadata columns for model training."""
    prop = prop.lower()
    return [
        *_GOLD_KEYS,
        *_META_COLS,
        *ML_PROP_FEATURES[prop],
        ML_PROP_TARGETS[prop],
    ]


def _latest_model_path(prop: str, models_dir: Path) -> Path:
    files = sorted(models_dir.glob(f"{prop}_quantile_xgb*.joblib"))
    if not files:
        raise FileNotFoundError(
            f"No saved model for prop '{prop}' in {models_dir}. "
            f"Run scripts/train_model.py --prop {prop} first."
        )
    return files[-1]


def load_model_bundle(
    prop: str,
    *,
    model_path: str | Path | None = None,
    models_dir: str | Path = _DEFAULT_MODELS_DIR,
) -> dict:
    """Load a quantile model bundle saved by ``train_prop_model``."""
    import joblib

    prop = prop.lower()
    path = Path(model_path) if model_path else _latest_model_path(prop, Path(models_dir))
    bundle = joblib.load(path)
    bundle["model_path"] = str(path)
    return bundle


def _prepare_training_frame(prop: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prop = prop.lower()
    feature_names = ML_PROP_FEATURES[prop]
    target_col = ML_PROP_TARGETS[prop]

    if "GAME_DATE" in df.columns:
        df = df.sort_values("GAME_DATE")

    filt = _TRAINING_FILTERS[prop]
    work = df.loc[filt(df)].copy()

    X = align_feature_frame(work, feature_names)
    target_key = _canonical_col(target_col)
    if target_col in work.columns:
        y = pd.to_numeric(work[target_col], errors="coerce")
    elif target_key in work.columns:
        y = pd.to_numeric(work[target_key], errors="coerce")
    else:
        raise ValueError(f"Target column {target_col!r} not found in training frame")

    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]


def _time_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(X) < 20:
        raise ValueError(f"Need at least 20 training rows, got {len(X)}")

    split_idx = max(1, int(len(X) * (1.0 - val_fraction)))
    split_idx = min(split_idx, len(X) - 1)
    return (
        X.iloc[:split_idx],
        y.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[split_idx:],
    )


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, object]:
    """Train one XGB quantile regressor per quantile level."""
    from xgboost import XGBRegressor

    models: dict[str, object] = {}
    for q in ML_QUANTILES:
        model = XGBRegressor(**ML_XGB_PARAMS, quantile_alpha=q)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        models[f"q_{q:.2f}"] = model
    return models


def _compute_val_metrics(
    models: dict[str, object],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Compute pinball loss per quantile and MAE on the P50 over the held-out split."""
    metrics: dict[str, float] = {}
    for key, model in models.items():
        q = float(key.split("_", 1)[1])
        y_pred = model.predict(X_val)
        errors = y_val.values - y_pred
        pinball = float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))
        metrics[f"pinball_{key}"] = round(pinball, 4)

    if "q_0.50" in models:
        y_pred_median = models["q_0.50"].predict(X_val)
        mae = float(np.mean(np.abs(y_val.values - y_pred_median)))
        metrics["mae_p50"] = round(mae, 4)

    return metrics


def train_prop_model(
    prop: str,
    *,
    season_year: str | None = None,
    season_type: str = "Regular Season",
    val_fraction: float = 0.15,
    models_dir: str | Path = _DEFAULT_MODELS_DIR,
    save_path: str | Path | None = None,
) -> Path:
    """Train a quantile model from ``ml.features_*``, register it, and save a joblib bundle.

    Each call generates a fresh UUID, inserts a row into ``ml.model_registry``,
    and saves the bundle as ``{prop}_{model_id}.joblib``.  The new entry is
    automatically made the active model for that prop_type.
    """
    import joblib

    from src.utils.db import insert_model_registry

    prop = prop.lower()
    df = read_ml_features(prop, season_year=season_year, season_type=season_type)
    if df.empty:
        raise ValueError(f"No rows found in ml.{ML_PROP_TABLES[prop]}")

    X, y = _prepare_training_frame(prop, df)
    X_train, y_train, X_val, y_val = _time_split(X, y, val_fraction=val_fraction)
    models = fit_quantile_models(X_train, y_train, X_val, y_val)

    val_metrics = _compute_val_metrics(models, X_val, y_val)

    feature_names = ML_PROP_FEATURES[prop]
    feat_hash = hashlib.sha256(json.dumps(sorted(feature_names)).encode()).hexdigest()[:8]
    feature_set_version = f"v_{feat_hash}"

    training_season: str | None = None
    if "SEASON_YEAR" in df.columns and df["SEASON_YEAR"].notna().any():
        seasons = sorted(str(s) for s in df["SEASON_YEAR"].dropna().unique())
        training_season = ",".join(seasons)

    train_end = None
    if "GAME_DATE" in df.columns and df["GAME_DATE"].notna().any():
        train_end = pd.to_datetime(df["GAME_DATE"], errors="coerce").max()

    model_id = str(uuid.uuid4())
    saved_at = datetime.now(timezone.utc)

    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = out_dir / f"{prop}_{model_id}.joblib"
    else:
        save_path = Path(save_path)

    bundle = {
        "prop": prop,
        "model_id": model_id,
        "quantile_models": models,
        "feature_names": feature_names,
        "target": ML_PROP_TARGETS[prop],
        "feature_set_version": feature_set_version,
        "training_season": training_season,
        "validation_metrics": val_metrics,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "train_end": train_end,
        "saved_at": saved_at,
    }
    joblib.dump(bundle, save_path)

    insert_model_registry(
        model_id=model_id,
        prop_type=prop,
        trained_at=saved_at,
        feature_set_version=feature_set_version,
        training_season=training_season,
        validation_metrics=val_metrics,
        joblib_path=str(save_path),
    )

    return save_path


def _pick_column(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
        key = _canonical_col(name)
        if key in df.columns:
            return df[key]
    return None


def predict_quantiles(
    prop: str,
    features_df: pd.DataFrame,
    *,
    model_path: str | Path | None = None,
    models_dir: str | Path = _DEFAULT_MODELS_DIR,
) -> pd.DataFrame:
    """Run saved quantile models on a feature frame."""
    prop = prop.lower()
    bundle = load_model_bundle(prop, model_path=model_path, models_dir=models_dir)
    feature_names = bundle["feature_names"]
    models = bundle["quantile_models"]

    X = align_feature_frame(features_df, feature_names)
    valid = X.notna().all(axis=1)
    if not valid.any():
        raise ValueError(f"No complete feature rows available for prop '{prop}'")

    X_valid = X.loc[valid]
    out = pd.DataFrame(index=X_valid.index)
    for out_name, candidates in (
        ("GAME_ID", ("GAME_ID", "game_id")),
        ("PLAYER_ID", ("PLAYER_ID", "player_id")),
        ("GAME_DATE", ("GAME_DATE", "game_date")),
        ("PLAYER_NAME", ("PLAYER_NAME", "player_name")),
    ):
        col = _pick_column(features_df.loc[valid], *candidates)
        if col is not None:
            out[out_name] = col.values

    for key, model in models.items():
        out[key.upper()] = model.predict(X_valid)

    out["PREDICTION"] = out["Q_0.50"]
    out["PROP"] = prop
    out["MODEL_PATH"] = bundle.get("model_path")
    out["MODEL_ID"] = bundle.get("model_id")
    out["PREDICTED_AT"] = datetime.now(timezone.utc)
    return out.reset_index(drop=True)


def prepare_predictions_upload(predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction rows for ``ml.predictions`` upsert."""
    required = {"PROP", "GAME_ID", "PLAYER_ID", "PREDICTION", "PREDICTED_AT"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions frame missing columns: {sorted(missing)}")

    out = predictions.copy()
    out["prop"] = out["PROP"].astype(str).str.lower()
    out["game_id"] = out["GAME_ID"].astype(str)
    out["player_id"] = pd.to_numeric(out["PLAYER_ID"], errors="coerce").astype("Int64")
    out["prediction"] = pd.to_numeric(out["PREDICTION"], errors="coerce")
    out["predicted_at"] = pd.to_datetime(out["PREDICTED_AT"], utc=True, errors="coerce")

    if "GAME_DATE" in out.columns:
        out["game_date"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.date
    elif "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    else:
        out["game_date"] = None

    if "PLAYER_NAME" in out.columns:
        out["player_name"] = out["PLAYER_NAME"]
    elif "player_name" not in out.columns:
        out["player_name"] = None

    if "MODEL_PATH" in out.columns:
        out["model_path"] = out["MODEL_PATH"].astype(str)
    elif "model_path" not in out.columns:
        out["model_path"] = None

    if "MODEL_ID" in out.columns:
        out["model_id"] = out["MODEL_ID"].astype(str).where(out["MODEL_ID"].notna(), None)
    elif "model_id" not in out.columns:
        out["model_id"] = None

    keep = [
        "prop",
        "game_id",
        "player_id",
        "prediction",
        "predicted_at",
        "game_date",
        "player_name",
        "model_path",
        "model_id",
    ]
    out = out[keep]
    out = out[out["prediction"].notna() & out["player_id"].notna()]
    return out.reset_index(drop=True)
