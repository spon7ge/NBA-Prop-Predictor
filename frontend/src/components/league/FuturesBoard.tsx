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
    <div
      className="space-y-4"
      aria-label="Loading futures"
    >
      {Array.from({ length: 1 }, (_, i) => (
        <div
          key={i}
          className="h-72 animate-pulse rounded-xl border border-white/10 bg-white/[0.03]"
        />
      ))}
    </div>
  );
}

function MarketBlock({ market }: { market: ApiWnbaFuturesMarket }) {
  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-base font-semibold tracking-tight text-white">
          {market.display_name}
        </h3>
        <p className="text-xs text-white/40">
          Odds by <span className="text-white/55">{market.provider}</span>
        </p>
      </div>
      {market.entries.length === 0 ? (
        <p className="text-sm text-white/40">No futures listed</p>
      ) : (
        <ul className="divide-y divide-white/5">
          {market.entries.map((entry) => (
            <li
              key={`${market.id}-${entry.team_id}`}
              className="flex items-center justify-between gap-3 py-2.5 first:pt-0 last:pb-0"
            >
              <div className="flex min-w-0 items-center gap-2.5">
                <TeamAbbrevAvatar
                  abbrev={entry.abbrev}
                  logoUrl={entry.logo_url}
                  sizeClassName="size-7"
                />
                <span className="truncate text-sm text-white/85">
                  {entry.name}
                </span>
              </div>
              <span className="shrink-0 font-mono text-sm font-semibold tabular-nums text-white">
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
    <section className="mx-auto max-w-6xl space-y-6 px-4 pb-16 sm:px-6 sm:pb-20">
      <header>
        <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          Futures
        </h2>
        <p className="mt-2 text-sm text-white/40">{season} season</p>
      </header>
      {isLoading ? (
        <Skeletons />
      ) : isError ? (
        <p className="text-sm text-white/40">Unable to load futures</p>
      ) : markets.length === 0 ? (
        <p className="text-sm text-white/40">No futures listed</p>
      ) : (
        <div className="mx-auto max-w-xl space-y-4">
          {markets.map((market) => (
            <MarketBlock key={market.id} market={market} />
          ))}
        </div>
      )}
    </section>
  );
}
