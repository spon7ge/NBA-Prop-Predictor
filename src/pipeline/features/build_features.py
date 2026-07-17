"""
Feature Engineering for NBA / WNBA Betting Models

Reads ``raw.{nba,wnba}_*`` tables from Supabase and engineers leakage-safe
player + team features for training.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd

from src.pipeline.clean import read_raw_tables
from src.pipeline.fetch import LEAGUES, LeagueKey

HALFLIVES = [5, 10, 20]
# Require a few prior games before EWM is defined (avoids 1-game "averages").
EWM_MIN_PERIODS = 3

# Leakage convention (player + team): shift(1) THEN ewm/expanding.
# Never ewm().shift(1) — same numbers here, but mixed order invites leaks on edits.

# Counting stats → divided by MIN before rolling (produces ``{stat}_per_min``).
PLAYER_BASE_PER_MIN_STATS = [
    'pts', 'fga', 'fg3a', 'fta',
    'oreb', 'dreb', 'reb', 'ast', 'tov', 'stl', 'blk', 'blka',
]
# Already rates / minutes — kept as-is.
PLAYER_BASE_LEVEL_STATS = [
    'min', 'fg_pct', 'fg3_pct', 'ft_pct', 'plus_minus'
]
PLAYER_BASE_STATS = PLAYER_BASE_LEVEL_STATS + [
    f'{s}_per_min' for s in PLAYER_BASE_PER_MIN_STATS
]

PLAYER_ADV_STATS = [
    'off_rating', 'def_rating', 'net_rating',
    'ast_pct', 'ast_ratio', 'oreb_pct', 'dreb_pct', 'reb_pct',
    'efg_pct', 'ts_pct', 'usg_pct', 'pace', 'poss', 'pie',
]

# Tracking cols after clean._prepare_tracking rename (spd/dist/orbc/…).
PLAYER_TRACK_PER_MIN_STATS = [
    'orbc', 'drbc', 'rbc', 'tchs', 'sast',
    'ftast', 'pass', 'cfga', 'ufga', 'dfga',
]
PLAYER_TRACK_LEVEL_STATS = [
    'minutes', 'spd', 'dist',
]
PLAYER_TRACK_STATS = PLAYER_TRACK_LEVEL_STATS + [
    f'{s}_per_min' for s in PLAYER_TRACK_PER_MIN_STATS
]

class FeatureEngineer:
    """Creates features for NBA or WNBA player-game training frames."""

    def __init__(
        self,
        raw_data_dir='../../data/raw',
        processed_data_dir='../../data/processed',
        season: str | None = None,
        season_type: str = 'Regular Season',
        league: LeagueKey | Literal['nba', 'wnba'] = 'nba',
    ):
        if league not in LEAGUES:
            raise ValueError(f"Unknown league: {league!r}")
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.league: LeagueKey = league  # type: ignore[assignment]
        self.season = season or LEAGUES[self.league].default_season
        self.season_type = season_type
        os.makedirs(processed_data_dir, exist_ok=True)

    def load_data(self):
        """Load all five raw tables from Supabase (``raw.nba_*`` or ``raw.wnba_*``)."""
        label = LEAGUES[self.league].label
        print(
            f"Loading raw {label} data from Supabase "
            f"({self.season} {self.season_type})..."
        )

        try:
            frames = read_raw_tables(
                self.season, self.season_type, league=self.league,
            )
            player_base_df = frames['player_base']
            player_adv_df = frames['player_adv']
            team_base_df = frames['team_base']
            team_adv_df = frames['team_adv']
            tracking_df = frames['start_positions']

            print(f"  Loaded {len(team_base_df):,} team base records")
            print(f"  Loaded {len(team_adv_df):,} team adv records")
            print(f"  Loaded {len(player_base_df):,} player base records")
            print(f"  Loaded {len(player_adv_df):,} player adv records")
            print(f"  Loaded {len(tracking_df):,} tracking records")

            return player_base_df, player_adv_df, team_base_df, team_adv_df, tracking_df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None, None, None

    # ── Player features ───────────────────────────────────────────────────────

    def create_player_base_features(self, player_base_df):
        """Leakage-safe EWM, season avg, and lag-1 for player_base per-min rates."""
        print("\nCreating player base features...")
        df = player_base_df.copy()
        if 'min' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['min']):
                df['min'] = self._parse_minutes(df['min'])
            # Same-game minutes label kept in the merged training frame (before *_per_min).
            df['minutes'] = df['min']
        df = self._add_per_min_stats(df, PLAYER_BASE_PER_MIN_STATS)
        return self._add_player_features(
            df, stat_cols=PLAYER_BASE_STATS, prefix='base',
        )

    def create_player_adv_features(self, player_adv_df):
        """Leakage-safe EWM, season avg, and lag-1 for player_adv stats."""
        print("\nCreating player adv features...")
        return self._add_player_features(
            player_adv_df, stat_cols=PLAYER_ADV_STATS, prefix='adv',
        )

    def create_player_tracking_features(
        self,
        tracking_df,
        player_base_df: pd.DataFrame | None = None,
    ):
        """Leakage-safe EWM, season avg, and lag-1 for tracking per-min rates.

        Tracking rows often lack ``game_date`` / ``season_year`` — pass
        ``player_base_df`` so those can be merged on ``game_id`` + ``player_id``.
        """
        print("\nCreating player tracking features...")
        df = tracking_df.copy()
        df = self._prepare_tracking_frame(df, player_base_df)
        df = self._add_per_min_stats(df, PLAYER_TRACK_PER_MIN_STATS)
        return self._add_player_features(
            df, stat_cols=PLAYER_TRACK_STATS, prefix='track',
        )

    @staticmethod
    def _parse_minutes(series: pd.Series) -> pd.Series:
        """Parse ``MM:SS`` (or numeric) minutes strings to float minutes."""
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)

        def _one(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            s = str(v).strip()
            if not s or s.lower() in ('nan', 'none', ''):
                return np.nan
            if ':' in s:
                parts = s.split(':')
                try:
                    if len(parts) == 2:
                        return float(parts[0]) + float(parts[1]) / 60.0
                    if len(parts) == 3:
                        return (
                            float(parts[0]) * 60
                            + float(parts[1])
                            + float(parts[2]) / 60.0
                        )
                except ValueError:
                    return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan

        return series.map(_one).astype(float)

    def _prepare_tracking_frame(
        self,
        df: pd.DataFrame,
        player_base_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Normalize minutes → float and attach game_date / season_year if needed."""
        if 'minutes' in df.columns:
            # Raw tracking stores MM:SS strings — must be float before EWM.
            df['minutes'] = self._parse_minutes(df['minutes'])
            if 'min' not in df.columns:
                df['min'] = df['minutes']
        elif 'min' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['min']):
                df['min'] = self._parse_minutes(df['min'])
        else:
            print("  ⚠ tracking has no minutes/min — per-min cols will be skipped")

        needs_date = 'game_date' not in df.columns
        needs_season = 'season_year' not in df.columns
        if needs_date or needs_season:
            if player_base_df is None:
                if needs_date:
                    raise ValueError(
                        "tracking_df has no game_date — pass player_base_df= so dates "
                        "can be merged for chronological EWM / season avg"
                    )
            else:
                keys = ['game_id', 'player_id']
                missing_keys = [k for k in keys if k not in player_base_df.columns]
                if missing_keys:
                    raise ValueError(
                        f"player_base_df missing join keys for tracking merge: {missing_keys}"
                    )
                cols = keys[:]
                if needs_date:
                    if 'game_date' not in player_base_df.columns:
                        raise ValueError(
                            "tracking_df has no game_date and player_base_df also lacks "
                            "game_date — cannot merge dates for chronological EWM"
                        )
                    cols.append('game_date')
                if needs_season:
                    if 'season_year' not in player_base_df.columns:
                        raise ValueError(
                            "tracking_df has no season_year and player_base_df also lacks "
                            "season_year — cannot merge season for season-to-date features"
                        )
                    cols.append('season_year')
                lookup = player_base_df[cols].drop_duplicates(subset=keys)
                df = df.merge(lookup, on=keys, how='left')
                print(f"  merged date/season from player_base → {len(df):,} rows")

        if needs_date and 'game_date' not in df.columns:
            raise ValueError("tracking_df still missing game_date after prepare")

        # load_data/_prepare_tracking renames position → start_position; restore name
        if 'position' not in df.columns and 'start_position' in df.columns:
            df['position'] = df['start_position']

        # starting = 1 if player has a non-empty position, else 0
        if 'position' in df.columns:
            pos = df['position'].astype(str).str.strip()
            df['starting'] = (
                pos.notna()
                & pos.ne('')
                & pos.str.lower().ne('nan')
                & pos.str.lower().ne('none')
            ).astype(int)
            n_starters = int(df['starting'].sum())
            print(f"  starting flag: {n_starters:,} / {len(df):,} starter rows")
        else:
            print("  ⚠ no position column — starting not set")
            df['starting'] = 0

        return df

    @staticmethod
    def _add_per_min_stats(df: pd.DataFrame, raw_cols: list[str]) -> pd.DataFrame:
        """Create ``{col}_per_min`` = col / MIN (0 minutes → NaN)."""
        if 'min' not in df.columns:
            print("  ⚠ no min column — skipping per-min conversion")
            return df
        min_denom = df['min'].replace(0, np.nan)
        for col in raw_cols:
            if col not in df.columns:
                continue
            df[f'{col}_per_min'] = df[col] / min_denom
        return df

    def _add_player_features(
        self,
        df: pd.DataFrame,
        stat_cols: list[str],
        prefix: str,
        halflives: list[int] | None = None,
    ) -> pd.DataFrame:
        """Add ``{prefix}_{stat}_ewm_hl{h}``, ``_season_avg``, and ``_lag1`` cols.

        Leakage convention: ``shift(1)`` then ``ewm`` / ``expanding`` (see module note).
        """
        halflives = halflives or HALFLIVES
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

        present = [c for c in stat_cols if c in df.columns]
        missing = [c for c in stat_cols if c not in df.columns]
        if missing:
            print(f"  ⚠ skipping missing cols: {missing}")

        g = df.groupby('player_id', sort=False)
        # Prior games only (0 on a player's first row) — lets the model distrust thin history.
        if prefix == 'base':
            df['games_played'] = g.cumcount()

        # Season-to-date expanding mean (exclude current game).
        if 'season_year' in df.columns:
            season_g = df.groupby(['player_id', 'season_year'], sort=False)
        else:
            season_g = g

        for stat in present:
            # shift first, then window — do not reverse this order.
            shifted = g[stat].shift(1)

            for hl in halflives:
                ewm = (
                    shifted.groupby(df['player_id'], sort=False)
                    .ewm(halflife=hl, min_periods=EWM_MIN_PERIODS)
                    .mean()
                )
                df[f'{prefix}_{stat}_ewm_hl{hl}'] = ewm.reset_index(level=0, drop=True)

            df[f'{prefix}_{stat}_lag1'] = shifted

            season_shifted = season_g[stat].shift(1)
            if 'season_year' in df.columns:
                season_keys = [df['player_id'], df['season_year']]
                drop_levels = [0, 1]
            else:
                season_keys = df['player_id']
                drop_levels = 0
            season_avg = (
                season_shifted.groupby(season_keys, sort=False)
                .expanding(min_periods=1)
                .mean()
            )
            df[f'{prefix}_{stat}_season_avg'] = season_avg.reset_index(
                level=drop_levels, drop=True,
            )

        print(
            f"  {prefix}: {len(present)} stats × "
            f"({len(halflives)} ewm + season_avg + lag1) on {len(df):,} rows"
        )
        return df

    # ── Team features ─────────────────────────────────────────────────────────

    def create_team_base_features(self, team_base_df):
        """Create exponential weighted averages and other team-level features.

        Leakage convention: shift(1) then ewm (same as player path).
        """
        print("\nCreating team features...")

        team_base_df = team_base_df.copy()
        team_base_df['game_date'] = pd.to_datetime(team_base_df['game_date'])
        team_base_df = team_base_df.sort_values(
            by=['team_id', 'game_date']
        ).reset_index(drop=True)

        g = team_base_df.groupby('team_id', sort=False)
        team_base_df['team_games_played'] = g.cumcount()

        stats_to_roll = [
            'pts', 'fga', 'fg3a', 'fta', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'blka',
        ]
        for stat in stats_to_roll:
            shifted = g[stat].shift(1)
            for hl in HALFLIVES:
                ewm = (
                    shifted.groupby(team_base_df['team_id'], sort=False)
                    .ewm(halflife=hl, min_periods=EWM_MIN_PERIODS)
                    .mean()
                )
                team_base_df[f'team_{stat}_ewm_hl{hl}'] = ewm.reset_index(
                    level=0, drop=True,
                )

        print("  Team base features created.")
        return team_base_df

    def create_team_adv_features(self, team_adv_df):
        """Create exponential weighted averages and other team-level features.

        Leakage convention: shift(1) then ewm (same as player path).
        """
        print("\nCreating team adv features...")

        team_adv_df = team_adv_df.copy()
        team_adv_df['game_date'] = pd.to_datetime(team_adv_df['game_date'])
        team_adv_df = team_adv_df.sort_values(
            by=['team_id', 'game_date']
        ).reset_index(drop=True)

        g = team_adv_df.groupby('team_id', sort=False)

        stats_to_roll = [
            'off_rating', 'def_rating', 'net_rating', 'ast_pct', 'oreb_pct',
            'dreb_pct', 'reb_pct', 'efg_pct', 'ts_pct', 'pace', 'poss', 'pie',
        ]
        for stat in stats_to_roll:
            shifted = g[stat].shift(1)
            for hl in HALFLIVES:
                ewm = (
                    shifted.groupby(team_adv_df['team_id'], sort=False)
                    .ewm(halflife=hl, min_periods=EWM_MIN_PERIODS)
                    .mean()
                )
                team_adv_df[f'team_{stat}_ewm_hl{hl}'] = ewm.reset_index(
                    level=0, drop=True,
                )

        print("  Team adv features created.")
        return team_adv_df

    def create_matchup_team_base_features(self, team_base_df):
        """Attach opponent team_base EWM features via same-game partner team_id.

        Expects ``create_team_base_features`` output (``team_*_ewm_hl*`` cols).
        Adds ``opp_team_id`` and ``opp_*_ewm_hl*`` columns for the other team
        in each ``game_id``.
        """
        print("\nCreating matchup team base features...")
        return self._attach_opp_ewm_features(team_base_df)

    def create_matchup_team_adv_features(self, team_adv_df):
        """Attach opponent team_adv EWM features via same-game partner team_id.

        Expects ``create_team_adv_features`` output (``team_*_ewm_hl*`` cols).
        """
        print("\nCreating matchup team adv features...")
        return self._attach_opp_ewm_features(team_adv_df)

    @staticmethod
    def _attach_opp_ewm_features(team_df: pd.DataFrame) -> pd.DataFrame:
        """Find the other team in each game and merge their EWM columns as opp_*."""
        df = team_df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['team_id', 'game_date']).reset_index(drop=True)

        # Pair each row with the other team_id that shares the same game_id.
        partners = df[['game_id', 'team_id']].rename(columns={'team_id': 'opp_team_id'})
        df = df.merge(partners, on='game_id', how='left')
        df = df.loc[df['team_id'] != df['opp_team_id']].copy()

        # Own-team cols are team_*_ewm_hl*; map → opp_*_ewm_hl* (drop team_ prefix).
        ewm_cols = [
            c for c in df.columns
            if c.startswith('team_') and '_ewm_hl' in c
        ]
        if not ewm_cols:
            print("  ⚠ no team_*_ewm_hl* columns found — run create_team_*_features first")
            return df

        opp_rename = {c: f'opp_{c[len("team_"):]}' for c in ewm_cols}
        opp = (
            df[['game_id', 'team_id', *ewm_cols]]
            .rename(columns={'team_id': 'opp_team_id', **opp_rename})
        )
        # opp frame is keyed by (game_id, opp_team_id) = that opponent's own row.
        df = df.merge(opp, on=['game_id', 'opp_team_id'], how='left')

        print(f"  Attached {len(ewm_cols)} opp EWM cols → {len(df):,} team-game rows")
        return df

    @staticmethod
    def _engineered_cols(df: pd.DataFrame) -> list[str]:
        """Columns produced by feature helpers (EWM / season / lag / per-min / opp / starting)."""
        out = []
        for c in df.columns:
            if c in ("starting", "opp_team_id", "games_played", "minutes") or c.endswith(
                "games_played"
            ):
                out.append(c)
            elif any(
                tok in c
                for tok in ("_ewm_hl", "_season_avg", "_lag1", "_per_min")
            ) or c.startswith("opp_"):
                out.append(c)
        return out

    def _merge_training_frame(
        self,
        player_base: pd.DataFrame,
        player_adv: pd.DataFrame,
        tracking: pd.DataFrame,
        team_base: pd.DataFrame,
        team_adv: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge player + team feature frames into one player-game training table."""
        id_cols = ["game_id", "player_id", "team_id", "player_name"]
        meta = [c for c in ("game_date", "season_year", "matchup") if c in player_base.columns]

        base_feat = self._engineered_cols(player_base)
        df = player_base[id_cols + meta + base_feat].copy()

        adv_feat = self._engineered_cols(player_adv)
        if adv_feat:
            df = df.merge(
                player_adv[["game_id", "player_id", *adv_feat]],
                on=["game_id", "player_id"],
                how="left",
            )

        track_feat = [c for c in self._engineered_cols(tracking) if c != "minutes"]
        if track_feat:
            df = df.merge(
                tracking[["game_id", "player_id", *track_feat]],
                on=["game_id", "player_id"],
                how="left",
            )

        team_base_feat = self._engineered_cols(team_base)
        if team_base_feat:
            df = df.merge(
                team_base[["game_id", "team_id", *team_base_feat]],
                on=["game_id", "team_id"],
                how="left",
            )

        team_adv_feat = [c for c in self._engineered_cols(team_adv) if c != "opp_team_id"]
        if team_adv_feat:
            df = df.merge(
                team_adv[["game_id", "team_id", *team_adv_feat]],
                on=["game_id", "team_id"],
                how="left",
                suffixes=("", "_team_adv"),
            )

        # Keep minutes immediately before same-game *_per_min rate columns.
        rest = [c for c in df.columns if c not in id_cols]
        if "minutes" in rest:
            rest = [c for c in rest if c != "minutes"]
            insert_at = next(
                (i for i, c in enumerate(rest) if c.endswith("_per_min")),
                len([c for c in ("game_date", "season_year", "matchup") if c in rest]),
            )
            rest.insert(insert_at, "minutes")
        ordered = id_cols + rest
        return df[ordered]

    def run(self) -> pd.DataFrame | None:
        """Build all features, merge to one player-game frame, save parquet."""
        player_base_df, player_adv_df, team_base_df, team_adv_df, tracking_df = (
            self.load_data()
        )
        if player_base_df is None:
            return None

        player_base_df = self.create_player_base_features(player_base_df)
        player_adv_df = self.create_player_adv_features(player_adv_df)
        tracking_df = self.create_player_tracking_features(
            tracking_df, player_base_df=player_base_df,
        )

        team_base_df = self.create_team_base_features(team_base_df)
        team_base_df = self.create_matchup_team_base_features(team_base_df)
        team_adv_df = self.create_team_adv_features(team_adv_df)
        team_adv_df = self.create_matchup_team_adv_features(team_adv_df)

        print("\nMerging feature tables...")
        training_df = self._merge_training_frame(
            player_base_df, player_adv_df, tracking_df, team_base_df, team_adv_df,
        )

        output_path = self.training_parquet_path()
        training_df.to_parquet(output_path, index=False)
        print(
            f"\nSaved {len(training_df):,} rows × {training_df.shape[1]} cols → {output_path}"
        )
        return training_df

    def training_parquet_path(self) -> str:
        """Path where ``run()`` writes this season's parquet."""
        season_type_slug = self.season_type.replace(' ', '_')
        name = f"{self.season}_{season_type_slug}_training_data.parquet"
        if self.league != 'nba':
            name = f"{self.league}_{name}"
        return os.path.join(self.processed_data_dir, name)


def main():
    """Main execution function"""
    print("Feature Engineering Script")
    print("=" * 60)

    engineer = FeatureEngineer()
    engineer.run()

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
