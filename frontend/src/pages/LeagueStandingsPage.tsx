import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { StandingsGrid } from "@/components/league/StandingsGrid";
import { useWnbaStandings } from "@/hooks/useWnbaStandings";

export function LeagueStandingsPage() {
  const { data, isLoading, hasNeverLoaded } = useWnbaStandings();
  const season = data?.season ?? new Date().getFullYear();

  return (
    <div className="space-y-0">
      <LeagueSubnav league="wnba" />
      <StandingsGrid
        season={season}
        conferences={data?.conferences ?? []}
        isLoading={isLoading && !data}
        isError={hasNeverLoaded}
      />
    </div>
  );
}
