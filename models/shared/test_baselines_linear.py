"""Tests for quantile linear baseline helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _toy_frame(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    features = ["starting", "base_min_season_avg", "base_min_lag1"]
    starting = rng.integers(0, 2, size=n)
    season_avg = rng.uniform(10, 35, size=n)
    lag1 = season_avg + rng.normal(0, 3, size=n)
    minutes = (
        8.0
        + 10.0 * starting
        + 0.55 * season_avg
        + 0.15 * lag1
        + rng.normal(0, 2.0, size=n)
    )
    lag1 = lag1.astype(float)
    lag1[0] = np.nan
    lag1[7] = np.nan

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "starting": starting.astype(float),
        "base_min_season_avg": season_avg,
        "base_min_lag1": lag1,
        "minutes": minutes,
        "game_date": dates,
    })
    return df, features


class TestQuantileLinearBaseline(unittest.TestCase):
    def test_fit_quantile_linear_returns_three_models_and_ordered_preds(self):
        from models.shared.baselines import fit_quantile_linear

        df, features = _toy_frame()
        X = df[features]
        y = df["minutes"]
        cut = 80
        models, preds = fit_quantile_linear(
            X.iloc[:cut], y.iloc[:cut], X.iloc[cut:],
            quantiles=[0.10, 0.50, 0.90],
        )
        self.assertEqual(set(models), {"q_0.10", "q_0.50", "q_0.90"})
        self.assertEqual(set(preds), {"q_0.10", "q_0.50", "q_0.90"})
        n_val = len(X) - cut
        for key in ("q_0.10", "q_0.50", "q_0.90"):
            self.assertEqual(preds[key].shape, (n_val,))
            self.assertTrue(np.isfinite(preds[key]).all())
        self.assertGreater(np.mean(preds["q_0.10"] <= preds["q_0.50"]), 0.9)
        self.assertGreater(np.mean(preds["q_0.50"] <= preds["q_0.90"]), 0.9)

    def test_run_quantile_linear_baseline_three_way_ladder(self):
        from models.shared.baselines import run_quantile_linear_baseline

        df, features = _toy_frame(n=200)
        train_df = df.iloc[:150].reset_index(drop=True)
        holdout_df = df.iloc[150:].reset_index(drop=True)
        n_tr = len(train_df)

        fold1_train = np.zeros(n_tr, dtype=bool)
        fold1_val = np.zeros(n_tr, dtype=bool)
        fold1_train[:80] = True
        fold1_val[80:110] = True
        fold2_train = np.zeros(n_tr, dtype=bool)
        fold2_val = np.zeros(n_tr, dtype=bool)
        fold2_train[20:120] = True
        fold2_val[120:150] = True

        wf_results = [
            {
                "fold": "WF fold 1",
                "mae": 4.0,
                "n": int(fold1_val.sum()),
                "train_mask": fold1_train,
                "val_mask": fold1_val,
            },
            {
                "fold": "WF fold 2",
                "mae": 3.8,
                "n": int(fold2_val.sum()),
                "train_mask": fold2_train,
                "val_mask": fold2_val,
            },
        ]

        X = train_df[features]
        y = train_df["minutes"]
        xgb_preds_ho = {
            "q_0.10": holdout_df["minutes"].values - 2,
            "q_0.50": holdout_df["minutes"].values + 0.1,
            "q_0.90": holdout_df["minutes"].values + 2,
        }
        xgb_preds_last = {
            "q_0.10": train_df.loc[fold2_val, "minutes"].values - 2,
            "q_0.50": train_df.loc[fold2_val, "minutes"].values + 0.2,
            "q_0.90": train_df.loc[fold2_val, "minutes"].values + 2,
        }

        out = run_quantile_linear_baseline(
            X, y, train_df,
            holdout_df=holdout_df,
            wf_results=wf_results,
            features=features,
            target_col="minutes",
            naive_primary="base_min_season_avg",
            xgb_preds_last=xgb_preds_last,
            xgb_preds_ho=xgb_preds_ho,
            last_fold=wf_results[-1],
            role_col="starting",
            alpha=0.05,
        )

        self.assertIn("wf_by_fold", out)
        self.assertEqual(len(out["wf_by_fold"]), 2)
        for row in out["wf_by_fold"]:
            self.assertIn("mae_naive", row)
            self.assertIn("mae_linear", row)
            self.assertIn("mae_xgb", row)
            self.assertTrue(np.isfinite(row["mae_linear"]))
            self.assertIn("coverage_80pct", row)

        self.assertIn("preds_ho", out)
        self.assertEqual(set(out["preds_ho"]), {"q_0.10", "q_0.50", "q_0.90"})
        self.assertEqual(len(out["preds_ho"]["q_0.50"]), len(holdout_df))

        self.assertIn("ho_metrics", out)
        self.assertTrue(np.isfinite(out["ho_metrics"]["mae"]))
        self.assertIn("coverage_80pct", out["ho_metrics"])

        for key in (
            "wf_linear_vs_naive",
            "wf_xgb_vs_linear",
            "ho_linear_vs_naive",
            "ho_xgb_vs_linear",
        ):
            self.assertIn(key, out)
        self.assertIn("reject_h0", out["ho_linear_vs_naive"])


if __name__ == "__main__":
    unittest.main()
