{# Map Odds-API / bookmaker full names to NBA tricodes. Pass through existing abbrevs. #}
{% macro standardize_team_abbrev(column) -%}
case upper(btrim({{ column }}::text))
    when 'ATLANTA HAWKS' then 'ATL'
    when 'BOSTON CELTICS' then 'BOS'
    when 'BROOKLYN NETS' then 'BKN'
    when 'NEW JERSEY NETS' then 'BKN'
    when 'CHARLOTTE HORNETS' then 'CHA'
    when 'CHARLOTTE BOBCATS' then 'CHA'
    when 'CHICAGO BULLS' then 'CHI'
    when 'CLEVELAND CAVALIERS' then 'CLE'
    when 'DALLAS MAVERICKS' then 'DAL'
    when 'DENVER NUGGETS' then 'DEN'
    when 'DETROIT PISTONS' then 'DET'
    when 'GOLDEN STATE WARRIORS' then 'GSW'
    when 'HOUSTON ROCKETS' then 'HOU'
    when 'INDIANA PACERS' then 'IND'
    when 'LA CLIPPERS' then 'LAC'
    when 'LOS ANGELES CLIPPERS' then 'LAC'
    when 'LOS ANGELES LAKERS' then 'LAL'
    when 'MEMPHIS GRIZZLIES' then 'MEM'
    when 'MIAMI HEAT' then 'MIA'
    when 'MILWAUKEE BUCKS' then 'MIL'
    when 'MINNESOTA TIMBERWOLVES' then 'MIN'
    when 'NEW ORLEANS PELICANS' then 'NOP'
    when 'NEW ORLEANS HORNETS' then 'NOP'
    when 'NEW YORK KNICKS' then 'NYK'
    when 'OKLAHOMA CITY THUNDER' then 'OKC'
    when 'ORLANDO MAGIC' then 'ORL'
    when 'PHILADELPHIA 76ERS' then 'PHI'
    when 'PHOENIX SUNS' then 'PHX'
    when 'PORTLAND TRAIL BLAZERS' then 'POR'
    when 'SACRAMENTO KINGS' then 'SAC'
    when 'SAN ANTONIO SPURS' then 'SAS'
    when 'TORONTO RAPTORS' then 'TOR'
    when 'UTAH JAZZ' then 'UTA'
    when 'WASHINGTON WIZARDS' then 'WAS'
    else upper(btrim({{ column }}::text))
end
{%- endmacro %}

{% macro standardize_market_category(column) -%}
case lower(btrim({{ column }}::text))
    when 'player_points' then 'points'
    when 'player_rebounds' then 'rebounds'
    when 'player_assists' then 'assists'
    when 'player_threes' then 'threes'
    when 'player_blocks' then 'blocks'
    when 'player_steals' then 'steals'
    when 'player_turnovers' then 'turnovers'
    when 'player_field_goals' then 'field_goals'
    when 'player_frees_made' then 'free_throws_made'
    when 'player_frees_attempts' then 'free_throws_attempted'
    when 'player_points_rebounds_assists' then 'pra'
    when 'player_points_rebounds' then 'pr'
    when 'player_points_assists' then 'pa'
    when 'player_rebounds_assists' then 'ra'
    when 'player_blocks_steals' then 'blocks_steals'
    else regexp_replace(lower(btrim({{ column }}::text)), '^player_', '')
end
{%- endmacro %}
