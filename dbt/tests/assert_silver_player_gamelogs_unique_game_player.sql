-- Test: no duplicate (game_id, player_id) rows in silver_player_gamelogs.
-- Returns rows when the test FAILS (non-empty result = failure).

select
    game_id,
    player_id,
    count(*) as row_count
from {{ ref('silver_player_gamelogs') }}
group by game_id, player_id
having count(*) > 1
