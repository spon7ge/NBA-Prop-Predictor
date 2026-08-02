import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { FuturesBoard } from "@/components/league/FuturesBoard";
import { useWnbaFutures } from "@/hooks/useWnbaFutures";

export function LeagueFuturesPage() {
  const { data, isLoading, hasNeverLoaded } = useWnbaFutures();

  return (
    <div className="space-y-0">
      <LeagueSubnav league="wnba" />
      <FuturesBoard
        season={data?.season ?? new Date().getFullYear()}
        markets={data?.markets ?? []}
        isLoading={isLoading && !data}
        isError={hasNeverLoaded}
      />
    </div>
  );
}
