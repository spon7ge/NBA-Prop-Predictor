"""Context / schedule / role features layered on FeatureEngineer output.

Operates on the merged player-game training frame (snake_case columns from
``build_features.FeatureEngineer``). All windows use prior games only.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.build_features import FeatureEngineer
from src.pipeline.fetch import LeagueKey

_PLAYER_POSITIONS_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "player_positions"
)
_FINE_POS: frozenset[str] = frozenset({"PG", "SG", "SF", "PF", "C"})
# Basketball-Reference labels used by ``wnba_*_players.csv``.
_WNBA_POS: frozenset[str] = frozenset(
    {"G", "F", "C", "G-F", "F-G", "F-C", "C-F"}
) | _FINE_POS


def _norm_player_name(name: object) -> str:
    """Lowercase + strip accents so CSV names match NBA API ``player_name``."""
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(n.lower().strip().split())


def _nba_season_end_year(season: str) -> str:
    """``'2024-25'`` → ``'2025'`` (suffix in ``nba_{end}_players.csv``)."""
    return str(int(str(season).split("-")[0]) + 1)


def _wnba_season_year(season: str) -> str:
    """``'2018'`` / ``2018`` → calendar year for ``wnba_{year}_players.csv``."""
    s = str(season).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return s


def _load_positions_lookup(
    season: str,
    *,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    """Load ``_name_lower`` → ``pos`` for one season from raw player_positions CSVs.

    * NBA  — ``nba_{end_year}_players.csv``, else ``player_positions.csv``
    * WNBA — ``wnba_{year}_players.csv`` (no NBA fallback)
    """
    if league == "wnba":
        year = _wnba_season_year(season)
        candidate = _PLAYER_POSITIONS_DIR / f"wnba_{year}_players.csv"
        path = candidate if candidate.exists() else None
        allowed = _WNBA_POS
    else:
        candidate = (
            _PLAYER_POSITIONS_DIR / f"nba_{_nba_season_end_year(season)}_players.csv"
        )
        path = (
            candidate
            if candidate.exists()
            else _PLAYER_POSITIONS_DIR / "player_positions.csv"
        )
        allowed = _FINE_POS

    if path is None or not path.exists():
        missing = candidate.name if league == "wnba" else f"season={season}"
        print(f"  ⚠ no positions CSV for {missing}")
        return pd.DataFrame(columns=["_name_lower", "pos"])

    raw = pd.read_csv(path, encoding="utf-8-sig")
    if "pos" not in raw.columns:
        print(f"  ⚠ {path.name} missing 'pos' — skipping")
        return pd.DataFrame(columns=["_name_lower", "pos"])

    name_cols = [c for c in ("name_s26", "name") if c in raw.columns]
    if not name_cols:
        print(f"  ⚠ {path.name} missing name columns — skipping")
        return pd.DataFrame(columns=["_name_lower", "pos"])

    # Keep hybrid labels like G-F intact (upper only letters/hyphen tokens).
    pos = raw["pos"].astype(str).str.strip().str.upper()
    valid = raw["pos"].notna() & pos.ne("") & pos.isin(allowed)
    parts = []
    for col in name_cols:
        keys = raw.loc[valid, col].map(_norm_player_name)
        part = pd.DataFrame({"_name_lower": keys, "pos": pos[valid]})
        parts.append(part[part["_name_lower"].ne("")])
    lookup = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset="_name_lower", keep="first"
    )
    print(f"  loaded {len(lookup):,} positions from {path.name} (season={season})")
    return lookup

# (short_ewm_col, long_ewm_col, output_col)
DEFAULT_TREND_5V20: list[tuple[str, str, str]] = [
    ("base_min_ewm_hl5", "base_min_ewm_hl20", "base_min_trend_5v20"),
    (
        "base_pts_per_min_ewm_hl5",
        "base_pts_per_min_ewm_hl20",
        "base_pts_per_min_trend_5v20",
    ),
    ("adv_ts_pct_ewm_hl5", "adv_ts_pct_ewm_hl20", "adv_ts_pct_trend_5v20"),
    ("adv_usg_pct_ewm_hl5", "adv_usg_pct_ewm_hl20", "adv_usg_pct_trend_5v20"),
    ("base_ast_per_min_ewm_hl5", "base_ast_per_min_ewm_hl20", "base_ast_per_min_trend_5v20"),
    ("base_oreb_per_min_ewm_hl5", "base_oreb_per_min_ewm_hl20", "base_oreb_per_min_trend_5v20"),
    ("base_dreb_per_min_ewm_hl5", "base_dreb_per_min_ewm_hl20", "base_dreb_per_min_trend_5v20"),
]

# (value_col already leakage-safe, output rank col)
DEFAULT_TEAM_RANKS: list[tuple[str, str]] = [
    ("base_pts_per_min_ewm_hl10", "team_pts_per_min_rank_l10"),
    ("base_ast_per_min_ewm_hl10", "team_ast_per_min_rank_l10"),
    ("base_oreb_per_min_ewm_hl10", "team_oreb_per_min_rank_l10"),
    ("base_dreb_per_min_ewm_hl10", "team_dreb_per_min_rank_l10"),
    ("adv_ts_pct_ewm_hl10", "team_ts_pct_rank_l10"),
    ("adv_usg_pct_ewm_hl10", "team_usg_pct_rank_l10"),
    ("base_min_ewm_hl10", "team_min_rank_l10"),
]

# Same-game raw cols → leakage-safe rolling std/var (shift then window).
DEFAULT_STD_WINDOWS: tuple[int, ...] = (5, 10, 15)
DEFAULT_STD_STATS: list[str] = [
    "minutes",
    "pts_per_min",
    "ast_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "reb_per_min",
    "ts_pct",
    "usg_pct",
]

def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _starter_features(
    df: pd.DataFrame,
    *,
    window: int = 10,
    min_periods: int = 3,
    starting_col: str = "starting",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Leakage-safe rolling start rate: mean of prior ``starting`` flags.

    ``starter_roll{window}_pct`` = share of the previous ``window`` games in
    which the player started (shift(1) then rolling mean).
    """
    df = df.copy()
    if "player_id" not in df.columns:
        raise ValueError("_starter_features requires player_id")
    if starting_col not in df.columns:
        raise ValueError(f"_starter_features: missing {starting_col}")
    if "game_date" not in df.columns:
        raise ValueError("_starter_features requires game_date")

    out_col = out_col or f"starter_roll{window}_pct"
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    shifted = df.groupby("player_id", sort=False)[starting_col].shift(1)
    df[out_col] = (
        shifted.groupby(df["player_id"], sort=False)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        .round(2)
    )
    return df


class ContextFeatureEngineer(FeatureEngineer):
    """FeatureEngineer + rest/B2B, trends, ranks, std/var, fatigue, starter flags."""

    def add_player_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge ``pos`` from ``data/raw/player_positions`` using ``self.league``.

        * NBA  — ``nba_{end_year}_players.csv`` (PG/SG/SF/PF/C);
                 falls back to ``player_positions.csv``.
        * WNBA — ``wnba_{year}_players.csv`` (G/F/C + hybrids).

        Uses row ``season_year`` / ``season`` when present, else ``self.season``.
        Matching uses accent-stripped ``player_name`` against CSV
        ``name_s26`` / ``name``.
        """
        print("\nMerging player positions...")
        if "player_name" not in df.columns:
            print("  ⚠ skip: missing player_name")
            return df

        df = df.copy()
        season_col = _resolve_col(df, ["season_year", "season"])
        if season_col is None:
            seasons = [self.season]
            print(f"  no season_year — using engineer season={self.season}")
        else:
            seasons = [
                s for s in df[season_col].dropna().unique().tolist() if s
            ] or [self.season]

        name_key = df["player_name"].map(_norm_player_name)
        pos = pd.Series(np.nan, index=df.index, dtype=object)
        for season in seasons:
            mask = (
                df[season_col] == season
                if season_col is not None
                else pd.Series(True, index=df.index)
            )
            if not mask.any():
                continue
            lookup = _load_positions_lookup(str(season), league=self.league)
            if lookup.empty:
                continue
            pos_map = lookup.set_index("_name_lower")["pos"]
            pos.loc[mask] = name_key.loc[mask].map(pos_map).to_numpy()

        df["pos"] = pos
        n_pos = int(df["pos"].notna().sum())
        print(
            f"  pos filled on {n_pos:,} / {len(df):,} rows "
            f"({len(seasons)} season file(s))"
        )
        return df

    def add_rest_and_b2b(
        self,
        df: pd.DataFrame,
        *,
        default_rest: float = 3.0,
    ) -> pd.DataFrame:
        """Add ``days_rest`` and ``is_back_to_back`` (days since prior player game)."""
        print("\nAdding rest / back-to-back...")
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

        df["days_rest"] = df.groupby("player_id", sort=False)["game_date"].diff().dt.days
        df["days_rest"] = df["days_rest"].fillna(default_rest)
        df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)

        n_b2b = int(df["is_back_to_back"].sum())
        print(f"  days_rest + is_back_to_back ({n_b2b:,} B2B rows)")
        return df

    def add_trend_5v20(
        self,
        df: pd.DataFrame,
        pairs: list[tuple[str, str, str]] | None = None,
    ) -> pd.DataFrame:
        """``short_ewm - long_ewm`` role/form trend (positive = recent uptick)."""
        print("\nAdding 5v20 trends...")
        df = df.copy()
        pairs = pairs or DEFAULT_TREND_5V20
        added = []
        for short_col, long_col, out_col in pairs:
            if short_col not in df.columns or long_col not in df.columns:
                print(f"  ⚠ skip {out_col}: need {short_col} and {long_col}")
                continue
            df[out_col] = df[short_col] - df[long_col]
            added.append(out_col)
        print(f"  added {len(added)} trend cols: {added}")
        return df

    def add_team_ranks(
        self,
        df: pd.DataFrame,
        ranks: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Dense rank of teammates on the same game by a prior-window metric.

        Example: among a team's players on ``game_date``, rank by
        ``base_pts_per_min_ewm_hl10`` (1 = highest). Value cols must already be
        leakage-safe (built with shift-then-window).
        """
        print("\nAdding within-team ranks...")
        df = df.copy()
        ranks = ranks or DEFAULT_TEAM_RANKS
        needed = {"team_id", "game_date"}
        if not needed.issubset(df.columns):
            raise ValueError(f"add_team_ranks requires columns {needed}")

        added = []
        for value_col, out_col in ranks:
            if value_col not in df.columns:
                print(f"  ⚠ skip {out_col}: missing {value_col}")
                continue
            df[out_col] = df.groupby(["team_id", "game_date"], sort=False)[
                value_col
            ].rank(ascending=False, method="dense")
            added.append(out_col)
        print(f"  added {len(added)} rank cols: {added}")
        return df

    def add_rolling_team_rank(
        self,
        df: pd.DataFrame,
        stat_col: str,
        *,
        window: int = 10,
        out_col: str | None = None,
        min_periods: int | None = None,
    ) -> pd.DataFrame:
        """Build leakage-safe ``stat`` roll{window}, then within-team-game rank.

        Matches the classic pattern:
        ``groupby(['team_id','game_date'])[f'{stat}_roll{window}'].rank(...)``.
        """
        print(f"\nAdding rolling team rank for {stat_col} (L{window})...")
        if stat_col not in df.columns:
            raise ValueError(f"add_rolling_team_rank: missing {stat_col}")
        if not {"player_id", "team_id", "game_date"}.issubset(df.columns):
            raise ValueError(
                "add_rolling_team_rank requires player_id, team_id, game_date"
            )

        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

        min_periods = min_periods if min_periods is not None else max(1, window // 2)
        roll_col = f"{stat_col}_roll{window}"
        shifted = df.groupby("player_id", sort=False)[stat_col].shift(1)
        df[roll_col] = (
            shifted.groupby(df["player_id"], sort=False)
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )

        out_col = out_col or f"team_{stat_col}_rank_l{window}"
        df[out_col] = df.groupby(["team_id", "game_date"], sort=False)[roll_col].rank(
            ascending=False, method="dense"
        )
        print(f"  {roll_col} → {out_col}")
        return df

    def add_fatigue_features(
        self,
        df: pd.DataFrame,
        *,
        min_col: str | None = None,
    ) -> pd.DataFrame:
        """Calendar-window fatigue proxies from prior games only (exclude current).

        Adds:
          - ``games_played_last_7_days``
          - ``games_played_last_14_days``
          - ``min_sum_last_7_days`` (requires a minutes column)
        """
        print("\nAdding fatigue window features...")
        if "player_id" not in df.columns or "game_date" not in df.columns:
            raise ValueError("add_fatigue_features requires player_id and game_date")

        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

        if min_col is None:
            for candidate in ("min", "minutes"):
                if candidate in df.columns:
                    min_col = candidate
                    break

        has_min = min_col is not None and min_col in df.columns
        if not has_min:
            print("  ⚠ no minutes col — min_sum_last_7_days will be NaN")

        games_7 = np.zeros(len(df), dtype=int)
        games_14 = np.zeros(len(df), dtype=int)
        min_sum_7 = np.full(len(df), np.nan if not has_min else 0.0, dtype=float)

        for idx in df.groupby("player_id", sort=False).groups.values():
            pos = df.index.get_indexer(idx)
            order = np.argsort(df.loc[idx, "game_date"].to_numpy())
            pos = pos[order]
            dates = df.loc[idx, "game_date"].to_numpy(dtype="datetime64[ns]")[order]
            mins = (
                df.loc[idx, min_col].to_numpy(dtype=float)[order]
                if has_min
                else None
            )

            for i in range(1, len(pos)):
                d = dates[i]
                prior_dates = dates[:i]
                mask_7 = prior_dates >= (d - np.timedelta64(7, "D"))
                mask_14 = prior_dates >= (d - np.timedelta64(14, "D"))
                games_7[pos[i]] = int(mask_7.sum())
                games_14[pos[i]] = int(mask_14.sum())
                if mins is not None:
                    min_sum_7[pos[i]] = float(np.nansum(mins[:i][mask_7]))

        df["games_played_last_7_days"] = games_7
        df["games_played_last_14_days"] = games_14
        df["min_sum_last_7_days"] = (
            np.round(min_sum_7, 1) if has_min else min_sum_7
        )
        print(
            "  games_played_last_7/14_days"
            + (f" + min_sum_last_7_days (from {min_col})" if has_min else "")
        )
        return df

    def add_games_played_to_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prior games played (career-in-frame and season-to-date).

        ``games_played`` / ``season_games_played`` are 0 on a player's first row
        (count of earlier games only).
        """
        print("\nAdding games_played to date...")
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

        df["games_played"] = df.groupby("player_id", sort=False).cumcount()
        if "season_year" in df.columns:
            df["season_games_played"] = df.groupby(
                ["player_id", "season_year"], sort=False
            ).cumcount()
            print("  games_played + season_games_played")
        else:
            print("  games_played (no season_year — season counter skipped)")
        return df

    def add_starter_features(
        self,
        df: pd.DataFrame,
        *,
        window: int = 10,
        min_periods: int = 3,
    ) -> pd.DataFrame:
        """Rolling prior-game start rate (``starter_roll{window}_pct``)."""
        print(f"\nAdding starter roll{window} features...")
        if "starting" not in df.columns:
            print("  ⚠ skip: missing starting column")
            return df
        out = _starter_features(df, window=window, min_periods=min_periods)
        col = f"starter_roll{window}_pct"
        n_ok = int(out[col].notna().sum())
        print(f"  {col} defined on {n_ok:,} / {len(out):,} rows")
        return out

    def add_rolling_std_var(
        self,
        df: pd.DataFrame,
        stat_cols: list[str] | None = None,
        windows: tuple[int, ...] | list[int] | None = None,
        *,
        min_periods: int | None = None,
        include_std: bool = True,
        include_var: bool = True,
    ) -> pd.DataFrame:
        """Leakage-safe rolling std / variance over prior games only.

        For each ``stat`` and ``window`` in ``windows`` (default 5/10/15), adds:
          - ``{stat}_roll{window}_std``
          - ``{stat}_roll{window}_var``

        Uses ``shift(1)`` then ``rolling`` so the current game is excluded.
        ``min`` falls back to ``minutes`` when only the latter is present.
        """
        if not include_std and not include_var:
            raise ValueError("add_rolling_std_var: enable include_std and/or include_var")
        if "player_id" not in df.columns or "game_date" not in df.columns:
            raise ValueError("add_rolling_std_var requires player_id and game_date")

        stat_cols = list(stat_cols or DEFAULT_STD_STATS)
        windows = tuple(windows or DEFAULT_STD_WINDOWS)

        print(f"\nAdding rolling std/var (windows={list(windows)})...")
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

        # Allow minutes alias for the default "min" slot.
        resolved: list[str] = []
        for col in stat_cols:
            if col in df.columns:
                resolved.append(col)
            elif col == "min" and "minutes" in df.columns:
                resolved.append("minutes")
                print("  using minutes in place of min")
            else:
                print(f"  ⚠ skip missing col: {col}")

        if not resolved:
            print("  ⚠ no stats available — skipping")
            return df

        added: list[str] = []
        for stat in resolved:
            shifted = df.groupby("player_id", sort=False)[stat].shift(1)
            grouped = shifted.groupby(df["player_id"], sort=False)
            for window in windows:
                mp = (
                    min_periods
                    if min_periods is not None
                    else max(2, window // 2)
                )
                roller = grouped.rolling(window, min_periods=mp)
                if include_std:
                    out_std = f"{stat}_roll{window}_std"
                    df[out_std] = (
                        roller.std()
                        .reset_index(level=0, drop=True)
                        .round(4)
                    )
                    added.append(out_std)
                if include_var:
                    out_var = f"{stat}_roll{window}_var"
                    df[out_var] = (
                        roller.var()
                        .reset_index(level=0, drop=True)
                        .round(4)
                    )
                    added.append(out_var)

        print(f"  added {len(added)} cols on {len(resolved)} stats")
        return df

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all context features to a merged training frame."""
        print("\n" + "=" * 60)
        print("Context features (pos, rest, trends, ranks, std/var, games played)")
        print("=" * 60)
        df = self.add_player_positions(df)
        df = self.add_games_played_to_date(df)
        df = self.add_rest_and_b2b(df)
        df = self.add_fatigue_features(df)
        df = self.add_trend_5v20(df)
        df = self.add_team_ranks(df)
        # Prefer ewm-based default ranks; also add classic roll10 rank if pts rate exists.
        if "base_pts_per_min" in df.columns:
            df = self.add_rolling_team_rank(
                df,
                "base_pts_per_min",
                window=10,
                out_col="team_pts_per_min_rank_roll10",
            )
        elif "pts_per_min" in df.columns:
            df = self.add_rolling_team_rank(
                df,
                "pts_per_min",
                window=10,
                out_col="team_pts_per_min_rank_roll10",
            )
        df = self.add_rolling_std_var(df)
        df = self.add_starter_features(df)
        return df

    def run(self) -> pd.DataFrame | None:
        """Build base FeatureEngineer frame, then layer context features and save."""
        training_df = super().run()
        if training_df is None:
            return None

        # Re-load avoided: enrich the in-memory merge (super already wrote parquet).
        training_df = self.enrich(training_df)

        season_type_slug = self.season_type.replace(" ", "_")
        output_path = os.path.join(
            self.processed_data_dir,
            f"{self.season}_{season_type_slug}_training_data.parquet",
        )
        training_df.to_parquet(output_path, index=False)
        print(
            f"\nSaved context-enriched {len(training_df):,} rows × "
            f"{training_df.shape[1]} cols → {output_path}"
        )
        return training_df


def main():
    print("Context Feature Engineering")
    print("=" * 60)
    ContextFeatureEngineer().run()
    print("\n" + "=" * 60)
    print("Context feature engineering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
