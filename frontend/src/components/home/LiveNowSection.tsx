import type { HomeLeague, LiveGame } from "./types";
import {
  LIVE_NOW_SKELETON_COUNT,
  formatGamesInProgress,
  normalizeLiveGames,
} from "./format";
import { SectionHeading } from "./SectionHeading";

type LiveNowSectionProps = {
  games?: LiveGame[];
  isLoading?: boolean;
};

const leaguePill: Record<HomeLeague, string> = {
  nba: "bg-sky-600/90 text-white",
  wnba: "bg-violet-600/90 text-white",
};

function SkeletonGameCard() {
  return (
    <article
      className="rounded-xl border border-white/10 bg-[#141414] p-4"
      aria-hidden
    >
      <div className="mb-4 flex items-center justify-between">
        <span className="h-5 w-12 animate-pulse rounded-full bg-white/10" />
        <span className="flex items-center gap-2">
          <span className="size-1.5 rounded-full bg-red-500/50" />
          <span className="h-3 w-14 animate-pulse rounded bg-white/10" />
        </span>
      </div>
      <div className="space-y-3">
        {[0, 1].map((row) => (
          <div key={row} className="flex items-center gap-3">
            <span className="size-7 animate-pulse rounded-full bg-white/10" />
            <div className="min-w-0 flex-1 space-y-1.5">
              <span className="block h-3 w-10 animate-pulse rounded bg-white/10" />
              <span className="block h-3 w-28 animate-pulse rounded bg-white/10" />
            </div>
            <span className="size-8 animate-pulse rounded-md bg-amber-400/15" />
          </div>
        ))}
      </div>
    </article>
  );
}

function LiveGameCard({ game }: { game: LiveGame }) {
  const inProgress = game.status === "live" || game.status === "halftime";

  return (
    <article className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <div className="mb-4 flex items-center justify-between">
        <span
          className={`rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase ${leaguePill[game.league]}`}
        >
          {game.league}
        </span>
        <span
          className={`flex items-center gap-2 text-xs ${
            inProgress ? "text-red-400" : "text-white/55"
          }`}
        >
          {inProgress ? (
            <span className="size-1.5 animate-pulse rounded-full bg-red-500" />
          ) : null}
          {game.statusLabel}
        </span>
      </div>
      <div className="space-y-3">
        {[game.away, game.home].map((team) => (
          <div key={team.abbrev} className="flex items-center gap-3">
            <span className="flex size-7 items-center justify-center rounded-full bg-white/10 text-[10px] font-bold text-white/70">
              {team.abbrev.slice(0, 1)}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold text-white">{team.abbrev}</p>
              <p className="truncate text-xs text-white/45">{team.name}</p>
            </div>
            <span className="flex size-8 items-center justify-center rounded-md bg-black font-mono text-sm font-bold text-amber-300">
              {team.score ?? "–"}
            </span>
          </div>
        ))}
      </div>
    </article>
  );
}

export function LiveNowSection({
  games,
  isLoading = false,
}: LiveNowSectionProps) {
  const list = normalizeLiveGames(games);
  const inProgressCount = list.filter(
    (g) => g.status === "live" || g.status === "halftime",
  ).length;
  const showSkeletons = isLoading && list.length === 0;

  return (
    <section id="live-now" className="mx-auto max-w-6xl px-4 pb-10 sm:px-6">
      <SectionHeading
        title="Live Now"
        subtitle={formatGamesInProgress(inProgressCount)}
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {showSkeletons
          ? Array.from({ length: LIVE_NOW_SKELETON_COUNT }, (_, i) => (
              <SkeletonGameCard key={i} />
            ))
          : list.map((game) => <LiveGameCard key={game.id} game={game} />)}
      </div>
    </section>
  );
}
