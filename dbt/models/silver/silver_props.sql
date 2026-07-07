-- Player prop lines: normalized names, deduped to latest quote per natural key.

with bronze as (
    select * from {{ ref('bronze_player_props') }}
    where player_name is not null
      and btrim(player_name) <> ''
      and side in ('over', 'under')
),

enriched as (
    select
        b.bookmaker,
        {{ standardize_market_category('b.market_category') }} as market_category,
        b.player_name as player_name_raw,
        {{ normalize_player_name('b.player_name') }} as normalized_name,
        p.player_id,
        coalesce(p.player_name, b.player_name) as player_name,
        b.side,
        b.game_date,
        b.line,
        b.odds,
        b.last_update_at,
        b.data_pulled_at,
        b.fetched_at,
        b.prop_source
    from bronze as b
    left join {{ ref('silver_players') }} as p
        on {{ normalize_player_name('b.player_name') }} = p.normalized_name
),

ranked as (
    select
        *,
        row_number() over (
            partition by
                bookmaker,
                market_category,
                normalized_name,
                side,
                game_date,
                prop_source
            order by
                last_update_at desc nulls last,
                fetched_at desc nulls last,
                data_pulled_at desc nulls last
        ) as row_num
    from enriched
)

select
    bookmaker,
    market_category,
    player_id,
    player_name,
    player_name_raw,
    normalized_name,
    side,
    game_date,
    line,
    odds,
    last_update_at,
    data_pulled_at,
    fetched_at,
    prop_source
from ranked
where row_num = 1
