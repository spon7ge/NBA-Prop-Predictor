-- Each ML feature table must have one row per player-game.
{% set ml_models = ['features', 'features_min', 'features_ppm', 'features_rpm', 'features_apm'] %}

{% for model in ml_models %}
select
    '{{ model }}' as model_name,
    game_id,
    player_id,
    count(*) as row_count
from {{ ref(model) }}
group by 1, 2, 3
having count(*) > 1
{% if not loop.last %}union all{% endif %}
{% endfor %}
