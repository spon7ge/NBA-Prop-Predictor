-- ML prediction output table (Phase 8).
-- Populated by scripts/generate_predictions.py

CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.predictions (
    prop            TEXT        NOT NULL,
    game_id         TEXT        NOT NULL,
    player_id       BIGINT      NOT NULL,
    prediction      NUMERIC     NOT NULL,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_date       DATE,
    player_name     TEXT,
    model_path      TEXT,

    PRIMARY KEY (prop, game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_date ON ml.predictions (game_date);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_player ON ml.predictions (player_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_predicted_at ON ml.predictions (predicted_at);
