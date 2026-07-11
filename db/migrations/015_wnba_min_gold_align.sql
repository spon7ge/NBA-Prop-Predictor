-- Align gold.wnba_player_min_model with WNBA_MIN_FEATURES
-- (src.pipeline.features.wnba_min_features).
-- Safe to re-run: ADD IF NOT EXISTS / DROP IF EXISTS.

ALTER TABLE gold.wnba_player_min_model ADD COLUMN IF NOT EXISTS days_rest NUMERIC;

ALTER TABLE gold.wnba_player_min_model DROP COLUMN IF EXISTS spd_10_ewm;
ALTER TABLE gold.wnba_player_min_model DROP COLUMN IF EXISTS ast_pct_lag2;
ALTER TABLE gold.wnba_player_min_model DROP COLUMN IF EXISTS pie_lag2;
ALTER TABLE gold.wnba_player_min_model DROP COLUMN IF EXISTS top_player_active;
ALTER TABLE gold.wnba_player_min_model DROP COLUMN IF EXISTS active_stars_count;
