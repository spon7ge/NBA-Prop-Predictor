-- ML layer schema (Phase 7).
-- Tables are built and refreshed by dbt models in dbt/models/ml/.
-- Run once in Supabase SQL Editor before the first `dbt run --select ml`.

CREATE SCHEMA IF NOT EXISTS ml;
