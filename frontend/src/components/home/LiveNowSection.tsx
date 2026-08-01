import { Link } from "react-router-dom";
import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";
import type { LiveGame } from "./types";
import {
  LIVE_NOW_SKELETON_COUNT,
  formatGamesInProgress,
  normalizeLiveGames,
} from "./format";
import { isInProgressStatus } from "./mapScoreboard";
import { SectionHeading } from "./SectionHeading";

type LiveNowSectionProps = {
  games?: LiveGame[];
  isLoading?: boolean;
  /** Set only when the scoreboard has never loaded, so good data is never replaced. */
  isError?: boolean;
};

function SkeletonGameCard() {
  return (
    <article
      className="rounded-xl border border-white/10 bg-white/[0.03] p-4"
      aria-hidden
    >
      <div className="mb-4 flex items-center justify-between">
        <span className="h-4 w-10 animate-pulse rounded bg-white/10" />
        <span className="flex items-center gap-2">
          <span className="size-1.5 rounded-full bg-red-500/50" />
          <span className="h-3 w-14 animate-pulse rounded bg-white/10" />
        </span>
      </div>
      <div className="space-y-3">
        {[0, 1].map((row) => (
          <div key={row} className="flex items-center gap-3">
            <span className="size-7 shrink-0 animate-pulse rounded-full bg-white/10" />
            <span className="h-3 w-10 shrink-0 animate-pulse rounded bg-white/10" />
            <span className="h-3 min-w-0 flex-1 animate-pulse rounded bg-white/10" />
            <span className="h-5 w-8 shrink-0 animate-pulse rounded bg-white/10" />
          </div>
        ))}
      </div>
    </article>
  );
}

function LiveGameCard({ game }: { game: LiveGame }) {
  const card = (
    <>
      <div className="mb-4 flex items-center justify-between">
        <span className="text-[11px] font-medium tracking-wide text-white/40 uppercase">
          {game.league}
        </span>
        <span className="flex items-center gap-2 text-xs text-red-400">
          <span className="size-1.5 animate-pulse rounded-full bg-red-500" />
          {game.statusLabel}
        </span>
      </div>
      <div className="space-y-3">
        {[game.away, game.home].map((team) => (
          <div key={team.abbrev} className="flex items-center gap-2.5">
            <TeamAbbrevAvatar
              abbrev={team.abbrev}
              logoUrl={team.logoUrl}
              sizeClassName="size-7"
            />
            <span className="shrink-0 text-xs font-semibold text-white">
              {team.abbrev}
            </span>
            <span className="min-w-0 flex-1 truncate text-xs text-white/45">
              {team.name}
            </span>
            <span className="shrink-0 font-mono text-sm font-semibold tracking-tight text-white">
              {team.score ?? "–"}
            </span>
          </div>
        ))}
      </div>
    </>
  );

  const baseCardClassName =
    "rounded-xl border border-white/10 bg-white/[0.03] p-4";

  if (game.espnEventId) {
    return (
      <Link
        to={`/games/${game.espnEventId}`}
        className={`block ${baseCardClassName} transition-colors hover:border-white/20`}
      >
        {card}
      </Link>
    );
  }

  return <article className={baseCardClassName}>{card}</article>;
}

export function LiveNowSection({
  games,
  isLoading = false,
  isError = false,
}: LiveNowSectionProps) {
  const list = normalizeLiveGames(games).filter((g) =>
    isInProgressStatus(g.status),
  );
  const inProgressCount = list.length;
  const showSkeletons = isLoading && list.length === 0;
  const showError = isError && !isLoading && list.length === 0;

  return (
    <section
      id="live-now"
      className="mx-auto max-w-6xl border-t border-white/10 px-4 py-16 sm:px-6 sm:py-20"
    >
      <SectionHeading
        title="Live now"
        subtitle={formatGamesInProgress(inProgressCount)}
      />

      {showError ? (
        <p role="status" className="text-sm text-white/40">
          Unable to load scoreboard
        </p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {showSkeletons
            ? Array.from({ length: LIVE_NOW_SKELETON_COUNT }, (_, i) => (
                <SkeletonGameCard key={i} />
              ))
            : list.map((game) => <LiveGameCard key={game.id} game={game} />)}
        </div>
      )}
    </section>
  );
}
