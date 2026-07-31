import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { PropPicksTable } from "@/components/league/PropPicksTable";
import { useWnbaProps } from "@/hooks/useWnbaProps";

export function LeaguePropPicksPage() {
  const { data, isLoading, isError, isFetched } = useWnbaProps();
  const props = data?.props ?? [];
  const showError = isError && !data;
  const showLoading = isLoading && !isFetched;

  return (
    <div className="space-y-6 py-6">
      <LeagueSubnav league="wnba" />
      <PropPicksTable
        props={props}
        isLoading={showLoading}
        isError={showError || Boolean(data && props.length === 0 && data.error)}
      />
    </div>
  );
}
