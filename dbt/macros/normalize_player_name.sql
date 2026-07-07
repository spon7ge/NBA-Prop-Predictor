{# Match src.utils.silver._norm_player_name / generalized_best_bets_v2.normalize_name #}
{% macro normalize_player_name(column) -%}
regexp_replace(
    regexp_replace(
        regexp_replace(
            lower(
                btrim(
                    regexp_replace(
                        regexp_replace(
                            translate(
                                {{ column }}::text,
                                'ÀÁÂÃÄÅàáâãäåÒÓÔÕÖØòóôõöøÈÉÊËèéêëÇçÌÍÎÏìíîïÙÚÛÜùúûüÿÑñÝýŠšŽžćčđ',
                                'AAAAAAaaaaaaOOOOOOooooooEEEEeeeeeCcIIIIiiiiUUUUuuuuyNnYySsZzccd'
                            ),
                            '''', '', 'g'
                        ),
                        '[^a-z0-9 ]', ' ', 'g'
                    )
                )
            ),
            '\s+(jr|sr|ii|iii|iv|v)\.?$', '', 'i'
        ),
        '\s+', ' ', 'g'
    ),
    '^\s+|\s+$', '', 'g'
)
{%- endmacro %}
