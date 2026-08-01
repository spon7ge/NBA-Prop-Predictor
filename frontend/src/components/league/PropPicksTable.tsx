import { useEffect, useState, type ReactNode } from "react";
import type { ApiWnbaPropBookQuote, ApiWnbaPropLine } from "@/lib/api";
import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";
import {
  PROP_BOOK_OPTIONS,
  type PropBookKey,
} from "@/components/league/filterPropLines";

export const PROP_PICKS_PAGE_SIZE = 50;

type PropPicksTableProps = {
  props: ApiWnbaPropLine[];
  isLoading?: boolean;
  isError?: boolean;
  /** True when filters hid all rows (API still returned props). */
  filtersActive?: boolean;
  /** Override default empty / error copy when set. */
  emptyMessage?: string;
  /** When set, only these book columns are shown. Empty/undefined → all. */
  visibleBooks?: Set<string>;
  /** Epoch ms of last successful props API fetch (React Query dataUpdatedAt). */
  lastUpdatedAt?: number;
  toolbar?: ReactNode;
};

export function formatPropPicksUpdatedAt(ms: number): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(ms));
}

function formatAmericanOdds(odds: number): string {
  return odds > 0 ? `+${odds}` : `${odds}`.replace("-", "−");
}

function OddsPill({ quote }: { quote: ApiWnbaPropBookQuote | null }) {
  if (!quote) {
    return <span className="text-white/20">&nbsp;</span>;
  }
  const odds = quote.odds_american;
  return (
    <span className="inline-flex flex-col items-center rounded-md border border-white/10 bg-white/[0.03] px-2.5 py-1 font-mono leading-tight text-white/75">
      <span className="text-[11px] text-white/90">{quote.line}</span>
      {odds != null ? (
        <span className="text-[10px] text-white/45">
          {formatAmericanOdds(odds)}
        </span>
      ) : null}
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

export function PropPicksTable({
  props,
  isLoading = false,
  isError = false,
  filtersActive = false,
  emptyMessage,
  visibleBooks,
  lastUpdatedAt,
  toolbar,
}: PropPicksTableProps) {
  const [page, setPage] = useState(0);

  // Reset when the result set identity changes (not on every new array ref).
  const listSignature = `${props.length}:${props[0]?.player_name ?? ""}:${props[0]?.market_type ?? ""}:${props[0]?.side ?? ""}:${props[props.length - 1]?.player_name ?? ""}:${props[props.length - 1]?.market_type ?? ""}:${props[props.length - 1]?.side ?? ""}`;
  useEffect(() => {
    setPage(0);
  }, [listSignature]);

  const emptyCopy =
    emptyMessage ??
    (filtersActive && !isError
      ? "No props match these filters"
      : "Prop lines unavailable");

  const bookColumns =
    visibleBooks && visibleBooks.size > 0
      ? PROP_BOOK_OPTIONS.filter((b) => visibleBooks.has(b.key))
      : [...PROP_BOOK_OPTIONS];

  const columns = [
    "Player",
    "Team",
    "Stat",
    "O/U",
    "Model",
    "EV",
    ...bookColumns.map((b) => b.label),
  ];

  const minWidthClass =
    bookColumns.length <= 4
      ? "min-w-[48rem]"
      : bookColumns.length <= 8
        ? "min-w-[72rem]"
        : "min-w-[120rem]";

  const totalPages = Math.max(1, Math.ceil(props.length / PROP_PICKS_PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const start = safePage * PROP_PICKS_PAGE_SIZE;
  const pageRows = props.slice(start, start + PROP_PICKS_PAGE_SIZE);
  const end = start + pageRows.length;
  const showPager = props.length > PROP_PICKS_PAGE_SIZE;

  return (
    <section className="mx-auto max-w-6xl space-y-5 px-4 pb-16 sm:px-6 sm:pb-20">
      <div className="flex flex-wrap items-baseline gap-3">
        <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          Prop Picks
        </h2>
        {lastUpdatedAt ? (
          <p className="text-sm text-white/40">
            Last updated {formatPropPicksUpdatedAt(lastUpdatedAt)}
          </p>
        ) : null}
      </div>
      {toolbar}
      {isLoading ? (
        <Skeletons />
      ) : isError || props.length === 0 ? (
        <p className="text-sm text-white/40">{emptyCopy}</p>
      ) : (
        <>
          <div className="overflow-x-auto rounded-xl border border-white/10">
            <table
              className={`w-full ${minWidthClass} border-collapse text-left text-sm`}
            >
              <thead>
                <tr className="border-b border-white/10 text-[10px] font-medium tracking-[0.14em] text-white/35 uppercase">
                  {columns.map((col) => (
                    <th key={col} className="px-3 py-2.5 font-medium">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row) => (
                  <tr
                    key={`${row.player_name}-${row.market_type}-${row.side}`}
                    className="border-b border-white/10 text-white/90 last:border-b-0"
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
                    <td className="px-3 py-3 text-white/55">{row.stat}</td>
                    <td className="px-3 py-3 text-white/70">
                      <SideLabel side={row.side} />
                    </td>
                    <td className="px-3 py-3 text-white/20" />
                    <td className="px-3 py-3 text-white/20" />
                    {bookColumns.map((book) => (
                      <td key={book.key} className="px-3 py-3">
                        <OddsPill quote={row[book.key as PropBookKey]} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-white/40">
              Showing {start + 1}–{end} of {props.length}
            </p>
            {showPager ? (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={safePage <= 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  className="rounded-md border border-white/10 px-3 py-1 text-sm text-white/55 enabled:hover:text-white disabled:cursor-not-allowed disabled:opacity-35"
                >
                  Previous
                </button>
                <span className="text-xs text-white/35">
                  Page {safePage + 1} of {totalPages}
                </span>
                <button
                  type="button"
                  disabled={safePage >= totalPages - 1}
                  onClick={() =>
                    setPage((p) => Math.min(totalPages - 1, p + 1))
                  }
                  className="rounded-md border border-white/10 px-3 py-1 text-sm text-white/55 enabled:hover:text-white disabled:cursor-not-allowed disabled:opacity-35"
                >
                  Next
                </button>
              </div>
            ) : null}
          </div>
        </>
      )}
      {!isLoading && props.length > 0 ? (
        <p className="text-xs text-white/35">Odds by Parlay API</p>
      ) : null}
    </section>
  );
}
