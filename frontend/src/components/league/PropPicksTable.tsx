import type { ApiWnbaPropBookQuote, ApiWnbaPropLine } from "@/lib/api";
import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";

type PropPicksTableProps = {
  props: ApiWnbaPropLine[];
  isLoading?: boolean;
  isError?: boolean;
};

function formatAmericanOdds(odds: number): string {
  return odds > 0 ? `+${odds}` : `${odds}`.replace("-", "−");
}

function formatBookPill(quote: ApiWnbaPropBookQuote | null): string | null {
  if (!quote) return null;
  return `${quote.line} ${formatAmericanOdds(quote.odds_american)}`;
}

function OddsPill({ quote }: { quote: ApiWnbaPropBookQuote | null }) {
  const label = formatBookPill(quote);
  if (!label) {
    return <span className="text-white/20">&nbsp;</span>;
  }
  return (
    <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-2.5 py-1 font-mono text-[11px] text-white/75">
      {label}
    </span>
  );
}

function SideLabel({ side }: { side: string }) {
  const lower = side.toLowerCase();
  if (lower === "over") return "Over";
  if (lower === "under") return "Under";
  return side;
}

function Skeletons() {
  return (
    <div className="space-y-0" aria-label="Loading prop picks">
      {Array.from({ length: 6 }, (_, i) => (
        <div
          key={i}
          className="h-11 animate-pulse border-b border-white/10 bg-white/[0.03]"
        />
      ))}
    </div>
  );
}

const COLUMNS = [
  "Player",
  "Team",
  "Stat",
  "O/U",
  "Model",
  "O/U%",
  "EV",
  "FanDuel",
  "DraftKings",
] as const;

export function PropPicksTable({
  props,
  isLoading = false,
  isError = false,
}: PropPicksTableProps) {
  return (
    <section className="mx-auto max-w-6xl space-y-3 px-4 sm:px-6">
      <h2 className="text-lg font-semibold text-white">Prop Picks</h2>
      {isLoading ? (
        <Skeletons />
      ) : isError || props.length === 0 ? (
        <p className="text-sm text-white/50">Prop lines unavailable</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[56rem] border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-white/10 text-[10px] font-semibold tracking-[0.14em] text-white/35 uppercase">
                {COLUMNS.map((col) => (
                  <th key={col} className="px-3 py-2 font-semibold">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {props.map((row) => (
                <tr
                  key={`${row.player_name}-${row.market_type}-${row.side}`}
                  className="border-b border-white/10 text-white/90"
                >
                  <td className="px-3 py-3 font-medium text-white">
                    {row.player_name}
                  </td>
                  <td className="px-3 py-3">
                    {row.team_abbrev ? (
                      <TeamAbbrevAvatar
                        abbrev={row.team_abbrev}
                        logoUrl={row.logo_url}
                        sizeClassName="size-7"
                      />
                    ) : (
                      <span className="text-white/20">&nbsp;</span>
                    )}
                  </td>
                  <td className="px-3 py-3 text-white/70">{row.stat}</td>
                  <td className="px-3 py-3 text-violet-300">
                    <SideLabel side={row.side} />
                  </td>
                  <td className="px-3 py-3 text-white/20" />
                  <td className="px-3 py-3 text-white/20" />
                  <td className="px-3 py-3 text-white/20" />
                  <td className="px-3 py-3">
                    <OddsPill quote={row.fanduel} />
                  </td>
                  <td className="px-3 py-3">
                    <OddsPill quote={row.draftkings} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {!isLoading && props.length > 0 ? (
        <p className="text-xs text-white/35">Odds by FanDuel &amp; DraftKings</p>
      ) : null}
    </section>
  );
}
