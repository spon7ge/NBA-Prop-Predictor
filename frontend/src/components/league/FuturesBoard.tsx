import type { ApiWnbaFuturesMarket } from "@/lib/api";
import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";

type FuturesBoardProps = {
  season: number;
  markets: ApiWnbaFuturesMarket[];
  isLoading?: boolean;
  isError?: boolean;
};

function Skeletons() {
  return (
    <div aria-label="Loading futures">
      <div className="h-40 animate-pulse rounded-xl border border-white/10 bg-white/[0.03] sm:h-48" />
    </div>
  );
}

function MarketBlock({ market }: { market: ApiWnbaFuturesMarket }) {
  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.03] p-3 sm:p-4">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-sm font-semibold tracking-tight text-white sm:text-base">
          {market.display_name}
        </h3>
        <p className="text-[11px] text-white/40 sm:text-xs">
          Odds by <span className="text-white/55">{market.provider}</span>
        </p>
      </div>
      {market.entries.length === 0 ? (
        <p className="text-sm text-white/40">No futures listed</p>
      ) : (
        <ul className="grid grid-cols-1 gap-x-8 gap-y-0.5 sm:grid-cols-2">
          {market.entries.map((entry) => (
            <li
              key={`${market.id}-${entry.team_id}`}
              className="flex items-center justify-between gap-2 py-1.5"
            >
              <div className="flex min-w-0 items-center gap-2">
                <TeamAbbrevAvatar
                  abbrev={entry.abbrev}
                  logoUrl={entry.logo_url}
                  sizeClassName="size-6"
                />
                <span className="truncate text-xs text-white/85 sm:text-sm">
                  {entry.name}
                </span>
              </div>
              <span className="shrink-0 font-mono text-xs font-semibold tabular-nums text-white sm:text-sm">
                {entry.odds_american}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function FuturesBoard({
  season,
  markets,
  isLoading = false,
  isError = false,
}: FuturesBoardProps) {
  return (
    <section className="mx-auto max-w-6xl space-y-4 px-4 pb-12 sm:space-y-5 sm:px-6 sm:pb-16">
      <header>
        <h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">
          Futures
        </h2>
        <p className="mt-1 text-xs text-white/40 sm:text-sm">{season} season</p>
      </header>
      {isLoading ? (
        <Skeletons />
      ) : isError ? (
        <p className="text-sm text-white/40">Unable to load futures</p>
      ) : markets.length === 0 ? (
        <p className="text-sm text-white/40">No futures listed</p>
      ) : (
        <div className="space-y-3">
          {markets.map((market) => (
            <MarketBlock key={market.id} market={market} />
          ))}
        </div>
      )}
    </section>
  );
}
