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
  if (league === "wnba") {
    return <WnbaMatchupsPage />;
  }

  return (
    <div className="space-y-6 py-6">
      <LeagueHero league="nba" />
      <LeagueSubnav league="nba" />
      <p className="mx-auto max-w-6xl px-4 text-sm text-white/50 sm:px-6">
        NBA matchups coming soon.
      </p>
    </div>
  );
}

function WnbaMatchupsPage() {
  const { games, isLoading, hasNeverLoaded, data } = useWnbaScoreboard();

  return (
    <div className="space-y-6 py-6">
      <LeagueHero league="wnba" dateEt={data?.date} />
      <LeagueSubnav league="wnba" />
      <MatchupsPanel
        games={mapToMatchupGames(games)}
        isLoading={isLoading}
        isError={hasNeverLoaded}
      />
    </div>
  );
}
