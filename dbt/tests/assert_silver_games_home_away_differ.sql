-- Test: home_team_abbrev must never equal away_team_abbrev in silver_games.
-- A team cannot play against itself.
-- Returns rows when the test FAILS (non-empty result = failure).

select
    game_id,
    game_date,
    home_team_abbrev,
    away_team_abbrev
from {{ ref('silver_games') }}
where home_team_abbrev = away_team_abbrev
