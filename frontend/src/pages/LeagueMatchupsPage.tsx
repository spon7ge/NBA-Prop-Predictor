import { LeagueHero } from "@/components/league/LeagueHero";
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { MatchupsPanel } from "@/components/league/MatchupsPanel";
import type { LeagueSlug } from "@/components/league/types";
import { mapToMatchupGames } from "@/components/home/mapScoreboard";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

type LeagueMatchupsPageProps = {
  league: LeagueSlug;
};

export function LeagueMatchupsPage({ league }: LeagueMatchupsPageProps) {
  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <LeagueHero league={league} />
      <LeagueSubnav league={league} />
      {league === "wnba" ? (
        <WnbaMatchupsBody />
      ) : (
        <p className="text-sm text-white/50">NBA matchups coming soon.</p>
      )}
    </div>
  );
}

function WnbaMatchupsBody() {
  const { games, isLoading, hasNeverLoaded } = useWnbaScoreboard();

  return (
    <MatchupsPanel
      games={mapToMatchupGames(games)}
      isLoading={isLoading}
      isError={hasNeverLoaded}
    />
  );
}
