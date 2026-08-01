import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { LeadersGrid } from "@/components/league/LeadersGrid";
import { useWnbaLeaders } from "@/hooks/useWnbaLeaders";

export function LeagueLeadersPage() {
  const { data, isLoading, hasNeverLoaded } = useWnbaLeaders();
  const season = data?.season ?? new Date().getFullYear();

  return (
    <div className="space-y-0">
      <LeagueSubnav league="wnba" />
      <LeadersGrid
        season={season}
        categories={data?.categories ?? []}
        isLoading={isLoading && !data}
        isError={hasNeverLoaded}
      />
    </div>
  );
}
