import type { ReactNode } from "react";
import { useParams } from "react-router-dom";
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { PlayerHeader } from "@/components/league/PlayerHeader";
import { PlayerRecentGames } from "@/components/league/PlayerRecentGames";
import { useWnbaPlayer } from "@/hooks/useWnbaPlayer";

function isNotFoundError(error: unknown): boolean {
  return error instanceof Error && /:\s*404\b/.test(error.message);
}

function PlayerPageSkeleton() {
  return (
    <div
      className="mx-auto max-w-6xl space-y-4 px-4 pb-16 sm:px-6 sm:pb-20"
      aria-label="Loading player"
    >
      <div className="h-28 animate-pulse rounded-xl border border-white/10 bg-white/[0.03]" />
      <div className="h-64 animate-pulse rounded-xl border border-white/10 bg-white/[0.03]" />
    </div>
  );
}

export function LeaguePlayerPage() {
  const { playerId = "" } = useParams<{ playerId: string }>();
  const { data, isLoading, hasNeverLoaded, error } = useWnbaPlayer(playerId);

  let body: ReactNode;
  if (isLoading && !data) {
    body = <PlayerPageSkeleton />;
  } else if (hasNeverLoaded && isNotFoundError(error)) {
    body = (
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <p className="text-sm text-white/60">Player not found</p>
      </div>
    );
  } else if (hasNeverLoaded || !data) {
    body = (
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <p className="text-sm text-white/60">Unable to load player</p>
      </div>
    );
  } else {
    body = (
      <div className="mx-auto max-w-6xl space-y-4 px-4 pb-16 sm:px-6 sm:pb-20">
        <PlayerHeader player={data} />
        <PlayerRecentGames games={data.games} />
        <p className="text-xs text-white/35">Data: stats.wnba.com</p>
      </div>
    );
  }

  return (
    <div className="space-y-0">
      <LeagueSubnav league="wnba" />
      {body}
    </div>
  );
}
