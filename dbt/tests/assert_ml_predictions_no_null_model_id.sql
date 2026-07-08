-- Every row in ml.predictions must reference a model_registry entry.
-- Rows written before the registry existed were backfilled with the legacy
-- placeholder (00000000-0000-0000-0000-000000000001) by migration 012, so
-- model_id should never be NULL.
--
-- dbt test convention: return rows on failure (test passes when result is empty).

select
    prop,
    game_id,
    player_id,
    predicted_at
from ml.predictions
where model_id is null
