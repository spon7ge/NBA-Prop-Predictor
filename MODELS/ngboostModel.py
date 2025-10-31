import numpy as np
import pandas as pd

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def fit_ngboost(train_df: pd.DataFrame,
                val_df: pd.DataFrame | None,
                features: list[str],
                target_col: str,
                learning_rate: float = 0.06,
                n_estimators: int = 500,
                max_depth: int = 3,
                random_state: int = 42,
                use_gpu: bool = False,
                base_max_depth: int | None = None):
    X_tr = train_df[features]
    y_tr = train_df[target_col].to_numpy()

    X_va = None
    y_va = None
    if val_df is not None and len(val_df) > 0:
        X_va = val_df[features]
        y_va = val_df[target_col].to_numpy()

    # Choose base learner (GPU via XGBoost, or CPU DecisionTree)
    if base_max_depth is None:
        base_max_depth = max_depth

    if use_gpu:
        base_est = XGBRegressor(
            max_depth=base_max_depth,
            tree_method='hist',
            device='cuda',
            n_estimators=1,  # single tree per stage as weak learner
            learning_rate=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            random_state=random_state,
            n_jobs=1
        )
    else:
        base_est = DecisionTreeRegressor(max_depth=base_max_depth)

    ngb = NGBRegressor(
        Dist=Normal,
        Score=MLE,
        natural_gradient=True,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=False,
        Base=base_est
    )

    if X_va is not None:
        ngb.fit(X_tr, y_tr, X_val=X_va, Y_val=y_va)
    else:
        ngb.fit(X_tr, y_tr)

    return ngb


def predict_mean_variance(model: NGBRegressor, df: pd.DataFrame, features: list[str]):
    """Return per-row mean and variance from NGBoost predictive distribution."""
    X = df[features]
    dist = model.pred_dist(X)  # Normal distribution per row
    mean = dist.loc
    var = dist.scale ** 2
    return mean, var


def predict_interval(model: NGBRegressor,
                     df: pd.DataFrame,
                     features: list[str],
                     alpha: float = 0.05):
    """Return lower/upper prediction intervals at 1-alpha."""
    X = df[features]
    dist = model.pred_dist(X)
    lower = dist.ppf(alpha / 2)
    upper = dist.ppf(1 - alpha / 2)
    return lower, upper


def _nll_score(estimator: NGBRegressor, X, y):
    """BayesSearchCV scorer: maximize negative log-likelihood (higher is better)."""
    dist = estimator.pred_dist(X)
    return float(-dist.logpdf(y).mean())


def chronological_split(df: pd.DataFrame,
                        date_col: str,
                        train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/val chronologically (oldest -> newest).

    - Ensures date_col is datetime
    - Sorts ascending by date
    - Returns (train_df, val_df)
    """
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    cut = int(len(data) * train_frac)
    train_df = data.iloc[:cut].reset_index(drop=True)
    val_df = data.iloc[cut:].reset_index(drop=True)
    return train_df, val_df


def fit_ngboost_bayes(train_df: pd.DataFrame,
                      val_df: pd.DataFrame | None,
                      features: list[str],
                      target_col: str,
                      n_iter: int = 30,
                      random_state: int = 42,
                      use_gpu: bool = False):
    """Bayesian hyperparameter search for NGBoost.

    Returns (best_estimator, best_params).
    """
    X_tr = train_df[features]
    y_tr = train_df[target_col].to_numpy()

    X_va = None
    y_va = None
    if val_df is not None and len(val_df) > 0:
        X_va = val_df[features]
        y_va = val_df[target_col].to_numpy()

    base_learn = XGBRegressor() if use_gpu else DecisionTreeRegressor()

    base_est = NGBRegressor(
        Dist=Normal,
        Score=MLE,
        natural_gradient=True,
        verbose=False,
        random_state=random_state,
        Base=base_learn
    )

    search_spaces = {
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'n_estimators': Integer(500, 2000),
        'minibatch_frac': Real(0.5, 1.0)
    }
    # Base learner hyperparameters to tune
    if use_gpu:
        search_spaces.update({
            'Base__max_depth': Integer(2, 8)
        })
    else:
        search_spaces.update({
            'Base__max_depth': Integer(2, 8)
        })

    search = BayesSearchCV(
        estimator=base_est,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring=_nll_score,
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    fit_params = {}
    if X_va is not None:
        fit_params = {'X_val': X_va, 'Y_val': y_va}

    search.fit(X_tr, y_tr, **fit_params)

    return search.best_estimator_, search.best_params_


