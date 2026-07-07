-- Test: no duplicate (game_id, player_id) rows in gold_player_game_stats.
-- Each player should appear exactly once per game.
-- Returns rows when the test FAILS (non-empty result = failure).

select
    game_id,
    player_id,
    player_name,
    count(*) as row_count
from {{ ref('gold_player_game_stats') }}
group by game_id, player_id, player_name
having count(*) > 1
