# ── FeatureEngineer: raw.{nba,wnba}_* → multi-season training frame ───────────
# Builds player (base/adv/tracking) + team (base/adv + opp matchup) features
# for each season, then concatenates into one player-game frame.
from src.pipeline.features.build_features import FeatureEngineer
from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Toggle: "nba" or "wnba"
LEAGUE = "wnba"
SEASON_TYPE = "Regular Season"
FORCE_REBUILD_FEATURES = True

SEASONS_BY_LEAGUE = {
    "nba": ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"],
    "wnba": ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"],
}
SEASONS = SEASONS_BY_LEAGUE[LEAGUE]

season_frames = []
for season in SEASONS:
    engineer = FeatureEngineer(
        season=season,
        season_type=SEASON_TYPE,
        league=LEAGUE,
        processed_data_dir=str(PROCESSED_DIR),
    )
    season_path = Path(engineer.training_parquet_path())

    if season_path.exists() and not FORCE_REBUILD_FEATURES:
        season_df = pd.read_parquet(season_path)
        print(f"Loaded {season_path.name}: {season_df.shape[0]:,} rows × {season_df.shape[1]} cols")
    else:
        season_df = engineer.run()
        if season_df is None:
            raise RuntimeError(f"FeatureEngineer returned no data for {LEAGUE} {season} {SEASON_TYPE}")
        print(f"Built {season_path.name}: {season_df.shape[0]:,} rows × {season_df.shape[1]} cols")

    season_frames.append(season_df)

feats = pd.concat(season_frames, ignore_index=True)
print(f"\nCombined feats ({LEAGUE}): {feats.shape[0]:,} rows × {feats.shape[1]} cols")
print(feats["season_year"].value_counts().sort_index())
feats.head()
