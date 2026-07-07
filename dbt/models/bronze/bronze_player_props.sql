-- Bronze player props: union DFS + US books with consistent names and types.

with dfs as (
    select
        bookmaker,
        category,
        name,
        over_under,
        commence_time,
        line,
        odds,
        last_update,
        data_pulled_at,
        fetched_at,
        'dfs'::text as prop_source
    from {{ source('raw', 'props_dfs') }}
),

us as (
    select
        bookmaker,
        category,
        name,
        over_under,
        commence_time,
        line,
        odds,
        last_update,
        data_pulled_at,
        fetched_at,
        'us'::text as prop_source
    from {{ source('raw', 'props_us') }}
),

combined as (
    select * from dfs
    union all
    select * from us
)

select
    bookmaker,
    category as market_category,
    name as player_name,
    lower(trim(over_under)) as side,
    case
        when commence_time ~ '^\d{4}-\d{2}-\d{2}$'
            then commence_time::date
        when commence_time ~ '^\d{4}-\d{2}-\d{2}T'
            then left(commence_time, 10)::date
        else null::date
    end as game_date,
    line::double precision as line,
    round(odds)::integer as odds,
    case
        when last_update ~ '^\d{4}-\d{2}-\d{2}T'
            then last_update::timestamptz
        when last_update ~ '^\d{4}-\d{2}-\d{2} '
            then replace(last_update, ' ', 'T')::timestamptz
        else null::timestamptz
    end as last_update_at,
    case
        when data_pulled_at ~ '^\d{4}-\d{2}-\d{2}T'
            then data_pulled_at::timestamptz
        when data_pulled_at ~ '^\d{4}-\d{2}-\d{2} '
            then replace(data_pulled_at, ' ', 'T')::timestamptz
        else null::timestamptz
    end as data_pulled_at,
    fetched_at,
    prop_source
from combined
