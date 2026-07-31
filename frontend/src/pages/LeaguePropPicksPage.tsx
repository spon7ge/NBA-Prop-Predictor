import { useState } from "react";
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { PropPicksFilters } from "@/components/league/PropPicksFilters";
import { PropPicksTable } from "@/components/league/PropPicksTable";
import {
  collectStatOptions,
  collectTeamOptions,
  filterPropLines,
} from "@/components/league/filterPropLines";
import { useWnbaProps } from "@/hooks/useWnbaProps";

export function LeaguePropPicksPage() {
  const { data, isLoading, isError, isFetched } = useWnbaProps();
  const props = data?.props ?? [];
  const showError = isError && !data;
  const showLoading = isLoading && !isFetched;
  const apiEmpty =
    showError || Boolean(data && props.length === 0 && data.error);

  const [selectedStats, setSelectedStats] = useState<Set<string>>(
    () => new Set(),
  );
  const [selectedSides, setSelectedSides] = useState<Set<string>>(
    () => new Set(),
  );
  const [selectedTeams, setSelectedTeams] = useState<Set<string>>(
    () => new Set(),
  );

  const filtersActive =
    selectedStats.size > 0 ||
    selectedSides.size > 0 ||
    selectedTeams.size > 0;

  const filtered = filterPropLines(props, {
    stats: selectedStats,
    sides: selectedSides,
    teams: selectedTeams,
  });

  return (
    <div className="space-y-6 py-6">
      <LeagueSubnav league="wnba" />
      <PropPicksTable
        props={filtered}
        isLoading={showLoading}
        isError={apiEmpty}
        filtersActive={filtersActive && !apiEmpty && props.length > 0}
        toolbar={
          !showLoading && !apiEmpty && props.length > 0 ? (
            <PropPicksFilters
              stats={collectStatOptions(props)}
              teams={collectTeamOptions(props)}
              selectedStats={selectedStats}
              selectedSides={selectedSides}
              selectedTeams={selectedTeams}
              onStatsChange={setSelectedStats}
              onSidesChange={setSelectedSides}
              onTeamsChange={setSelectedTeams}
              onClear={() => {
                setSelectedStats(new Set());
                setSelectedSides(new Set());
                setSelectedTeams(new Set());
              }}
            />
          ) : null
        }
      />
    </div>
  );
}
