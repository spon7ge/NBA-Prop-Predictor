import { ChevronLeft, ChevronRight } from "lucide-react";
import { isInProgressStatus } from "@/components/home/mapScoreboard";
import { MatchupGameCard } from "./MatchupGameCard";
import type { MatchupGame } from "./types";

type MatchupsPanelProps = {
  games: MatchupGame[];
  isLoading?: boolean;
  isError?: boolean;
};

function MatchupSkeletons() {
  return (
    <div
      className="grid grid-cols-1 gap-3 md:grid-cols-2"
      aria-label="Loading matchups"
    >
      {Array.from({ length: 3 }, (_, index) => (
        <div
          key={index}
          className="h-36 animate-pulse rounded-xl border border-white/10 bg-white/5"
        />
      ))}
    </div>
  );
}

function Section({
  label,
  games,
}: {
  label: string;
  games: MatchupGame[];
}) {
  return (
    <section className="space-y-4">
      <h3 className="text-[11px] font-medium tracking-[0.14em] text-white/35 uppercase">
        {label}
      </h3>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {games.map((game) => (
          <MatchupGameCard key={game.id} game={game} />
        ))}
      </div>
    </section>
  );
}

export function MatchupsPanel({
  games,
  isLoading = false,
  isError = false,
}: MatchupsPanelProps) {
  const live = games.filter((game) => isInProgressStatus(game.status));
  const rest = games.filter((game) => !isInProgressStatus(game.status));
  const gameLabel = games.length === 1 ? "game" : "games";

  return (
    <section className="mx-auto max-w-6xl space-y-8 px-4 pb-16 sm:px-6 sm:pb-20">
      <header>
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
            Matchups
          </h2>
          <div className="flex items-center gap-2">
            <button
              type="button"
              aria-label="Previous day"
              disabled
              className="flex size-8 items-center justify-center rounded-md border border-white/10 text-white/30 disabled:cursor-not-allowed"
            >
              <ChevronLeft aria-hidden="true" className="size-4" strokeWidth={1.75} />
            </button>
            <span className="min-w-14 text-center text-sm font-medium text-white/55">
              Today
            </span>
            <button
              type="button"
              aria-label="Next day"
              disabled
              className="flex size-8 items-center justify-center rounded-md border border-white/10 text-white/30 disabled:cursor-not-allowed"
            >
              <ChevronRight aria-hidden="true" className="size-4" strokeWidth={1.75} />
            </button>
          </div>
        </div>
        <p className="mt-2 text-sm text-white/40">
          {games.length} {gameLabel} · open a card for box score, play-by-play
          &amp; win probability
        </p>
      </header>

      {games.length === 0 ? (
        isLoading ? (
          <MatchupSkeletons />
        ) : (
          <p
            role={isError ? "status" : undefined}
            className="py-8 text-center text-sm text-white/40"
          >
            {isError
              ? "Unable to load matchups"
              : "No games on today's slate"}
          </p>
        )
      ) : (
        <div className="space-y-8">
          {live.length > 0 ? <Section label="Live now" games={live} /> : null}
          {rest.length > 0 ? (
            <Section label="Rest of the slate" games={rest} />
          ) : null}
        </div>
      )}
    </section>
  );
}
