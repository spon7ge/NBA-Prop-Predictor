-- Phase 12: Model Registry
--
-- 1. Creates ml.model_registry so every training run is recorded with its
--    validation metrics, feature set, and joblib path.
-- 2. Adds model_id FK to ml.predictions (nullable for backward compat).
-- 3. Inserts a 'legacy' placeholder entry and backfills all pre-registry rows.
--
-- Safe to run more than once (all DDL is IF NOT EXISTS / ON CONFLICT DO NOTHING).

CREATE SCHEMA IF NOT EXISTS ml;

-- ── ml.model_registry ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml.model_registry (
    model_id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    prop_type           TEXT        NOT NULL,
    trained_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    feature_set_version TEXT,
    training_season     TEXT,
    -- validation_metrics: { "mae_p50": float, "pinball_q_0.10": float, ... }
    validation_metrics  JSONB,
    joblib_path         TEXT,
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE
);

-- Fast lookup: which model is active for a given prop?
CREATE INDEX IF NOT EXISTS idx_model_registry_prop_active
    ON ml.model_registry (prop_type, is_active);

-- ── ml.predictions: add model_id FK ───────────────────────────────────────────

ALTER TABLE ml.predictions
    ADD COLUMN IF NOT EXISTS model_id UUID REFERENCES ml.model_registry(model_id);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_id
    ON ml.predictions (model_id);

-- ── Legacy placeholder ────────────────────────────────────────────────────────
-- One sentinel row covers all predictions written before the registry existed.
-- is_active = FALSE so it never surfaces as a "real" active model.

INSERT INTO ml.model_registry (
    model_id,
    prop_type,
    trained_at,
    feature_set_version,
    training_season,
    validation_metrics,
    joblib_path,
    is_active
)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'legacy',
    NOW(),
    'legacy',
    'legacy',
    '{"note": "placeholder for predictions written before the model registry was introduced"}'::jsonb,
    NULL,
    FALSE
)
ON CONFLICT (model_id) DO NOTHING;

-- ── Backfill existing predictions ─────────────────────────────────────────────

UPDATE ml.predictions
SET model_id = '00000000-0000-0000-0000-000000000001'
WHERE model_id IS NULL;
