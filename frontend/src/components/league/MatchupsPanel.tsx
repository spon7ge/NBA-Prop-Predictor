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
    <section className="space-y-3">
      <h3 className="text-xs font-semibold tracking-wider text-white/50">
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
    <section className="mx-auto max-w-6xl space-y-5 px-4 sm:px-6">
      <header>
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-xl font-semibold text-white">Matchups</h2>
          <div className="flex items-center gap-2">
            <button
              type="button"
              aria-label="Previous day"
              disabled
              className="flex size-8 items-center justify-center rounded-md border border-white/10 text-white/30 disabled:cursor-not-allowed"
            >
              <ChevronLeft aria-hidden="true" className="size-4" />
            </button>
            <span className="min-w-14 text-center text-sm font-medium text-white/70">
              Today
            </span>
            <button
              type="button"
              aria-label="Next day"
              disabled
              className="flex size-8 items-center justify-center rounded-md border border-white/10 text-white/30 disabled:cursor-not-allowed"
            >
              <ChevronRight aria-hidden="true" className="size-4" />
            </button>
          </div>
        </div>
        <p className="mt-1 text-sm text-white/45">
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
            className="py-8 text-center text-sm text-white/45"
          >
            {isError
              ? "Unable to load matchups"
              : "No games on today's slate"}
          </p>
        )
      ) : (
        <div className="space-y-6">
          {live.length > 0 ? <Section label="LIVE NOW" games={live} /> : null}
          {rest.length > 0 ? (
            <Section label="REST OF THE SLATE" games={rest} />
          ) : null}
        </div>
      )}
    </section>
  );
}
