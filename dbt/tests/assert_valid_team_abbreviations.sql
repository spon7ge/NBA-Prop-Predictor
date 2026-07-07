-- Test: team_abbreviation and opp_team_abbreviation in gold_player_game_stats
-- must be recognised NBA tricodes (current + recent historical).
-- Returns rows when the test FAILS (non-empty result = failure).

with valid_teams as (
    select unnest(array[
        -- Current 30 franchises (2025-26)
        'ATL', 'BOS', 'BKN', 'CHA', 'CHI',
        'CLE', 'DAL', 'DEN', 'DET', 'GSW',
        'HOU', 'IND', 'LAC', 'LAL', 'MEM',
        'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
        'OKC', 'ORL', 'PHI', 'PHX', 'POR',
        'SAC', 'SAS', 'TOR', 'UTA', 'WAS',
        -- Historical / alternate tricodes that may appear in older seasons
        'NJN', 'NOH', 'NOK', 'SEA', 'CHO',
        'VAN', 'WSB'
    ]) as abbrev
),

bad_team as (
    select game_id, player_id, team_abbreviation as abbrev, 'team' as column_name
    from {{ ref('gold_player_game_stats') }}
    where team_abbreviation is not null
      and team_abbreviation not in (select abbrev from valid_teams)
),

bad_opp as (
    select game_id, player_id, opp_team_abbreviation as abbrev, 'opp_team' as column_name
    from {{ ref('gold_player_game_stats') }}
    where opp_team_abbreviation is not null
      and opp_team_abbreviation not in (select abbrev from valid_teams)
)

select * from bad_team
union all
select * from bad_opp
