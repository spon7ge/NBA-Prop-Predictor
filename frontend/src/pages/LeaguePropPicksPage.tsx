import { useState } from "react";
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { PropPicksFilters } from "@/components/league/PropPicksFilters";
import { PropPicksTable } from "@/components/league/PropPicksTable";
import {
  collectStatOptions,
  collectTeamOptions,
  excludePastGameProps,
  filterPropLines,
} from "@/components/league/filterPropLines";
import { useWnbaProps } from "@/hooks/useWnbaProps";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function LeaguePropPicksPage() {
  const { data, isLoading, isError, isFetched, dataUpdatedAt } = useWnbaProps();
  const { games, data: scoreboard } = useWnbaScoreboard();
  const props = data?.props ?? [];
  const activeProps = excludePastGameProps(props, games, scoreboard?.date);
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
  const [selectedBooks, setSelectedBooks] = useState<Set<string>>(
    () => new Set(),
  );

  const filtersActive =
    selectedStats.size > 0 ||
    selectedSides.size > 0 ||
    selectedTeams.size > 0 ||
    selectedBooks.size > 0;

  const filtered = filterPropLines(activeProps, {
    stats: selectedStats,
    sides: selectedSides,
    teams: selectedTeams,
    books: selectedBooks,
  });

  return (
    <div className="space-y-6 py-6">
      <LeagueSubnav league="wnba" />
      <PropPicksTable
        props={filtered}
        isLoading={showLoading}
        isError={apiEmpty}
        visibleBooks={selectedBooks}
        lastUpdatedAt={dataUpdatedAt || undefined}
        filtersActive={filtersActive && !apiEmpty && activeProps.length > 0}
        toolbar={
          !showLoading && !apiEmpty && activeProps.length > 0 ? (
            <PropPicksFilters
              stats={collectStatOptions(activeProps)}
              teams={collectTeamOptions(activeProps)}
              selectedStats={selectedStats}
              selectedSides={selectedSides}
              selectedTeams={selectedTeams}
              selectedBooks={selectedBooks}
              onStatsChange={setSelectedStats}
              onSidesChange={setSelectedSides}
              onTeamsChange={setSelectedTeams}
              onBooksChange={setSelectedBooks}
              onClear={() => {
                setSelectedStats(new Set());
                setSelectedSides(new Set());
                setSelectedTeams(new Set());
                setSelectedBooks(new Set());
              }}
            />
          ) : null
        }
      />
    </div>
  );
}
