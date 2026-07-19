import { useState, type ReactNode } from "react";
import type { ApiLeagueFilter } from "@/types/api";
import type { EnrichedPick, PlayersGroupSort } from "@/types/slate";
import type { PlayersSortKey } from "@/lib/constants";
import {
  aggregateEnrichedByPlayer,
  filterByPlatform,
  filterPlayerRows,
  filterPlayerRowsByStat,
  sortPlayerGroups,
} from "@/lib/players";
import { useEnrichedPicks } from "@/lib/queries";
import { Dropdown } from "@/components/Dropdown";
import { PlayerBlock } from "@/components/PlayerBlock";
import {
  ErrorPanel,
  LoadingMessage,
  PlayersListSkeleton,
} from "@/components/LoadingSkeleton";

const LEAGUE_OPTIONS: { value: ApiLeagueFilter; label: string }[] = [
  { value: "wnba", label: "WNBA" },
  { value: "nba", label: "NBA" },
];

type PlatformFilter = "ALL" | "PrizePicks" | "Underdog" | "DraftKings Pick6" | "Betr DFS";

const PLATFORM_OPTIONS: { value: PlatformFilter; label: string }[] = [
  { value: "ALL", label: "All Platforms" },
  { value: "PrizePicks", label: "PrizePicks" },
  { value: "Underdog", label: "Underdog" },
  { value: "DraftKings Pick6", label: "DraftKings" },
  { value: "Betr DFS", label: "Betr" },
];

function FilterPills<T extends string>({
  items,
  active,
  onSelect,
  ariaLabel,
  id,
  getClassName,
}: {
  items: { value: T; label: string }[];
  active: T | null;
  onSelect: (value: T | null) => void;
  ariaLabel: string;
  id?: string;
  getClassName?: (value: T) => string;
}) {
  return (
    <div className="stat-filter" role="group" aria-label={ariaLabel} id={id}>
      {items.map((item) => {
        const isAll = item.value === ("ALL" as T);
        const on = isAll ? active == null : active === item.value;
        const extra = getClassName && !isAll ? getClassName(item.value) : "";
        return (
          <button
            key={item.value}
            type="button"
            className={`stat-pill${extra ? ` ${extra}` : ""}${on ? " active" : ""}`}
            aria-pressed={on}
            onClick={() => {
              if (isAll) onSelect(null);
              else onSelect(active === item.value ? null : item.value);
            }}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

function leagueBadgeLabel(league: ApiLeagueFilter): string {
  return league.toUpperCase();
}

interface AllPlayersViewProps {
  league: ApiLeagueFilter;
  onLeagueChange: (league: ApiLeagueFilter) => void;
}

export function AllPlayersView({ league, onLeagueChange }: AllPlayersViewProps) {
  const { data, isLoading, isError, error, refetch, isFetching } =
    useEnrichedPicks(league);
  const allEnriched = data?.picks ?? [];
  const dataSource = data?.source;
  const gameDate = data?.gameDate;

  const [search, setSearch] = useState("");
  const [activeStat, setActiveStat] = useState<string | null>(null);
  const [activePlatform, setActivePlatform] = useState<PlatformFilter>("ALL");
  const [expandedPlayer, setExpandedPlayer] = useState<string | null>(null);
  const [groupSort, setGroupSort] = useState<PlayersGroupSort>("edge_desc");
  const [sortKey, setSortKey] = useState<PlayersSortKey | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  function handleSort(key: PlayersSortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  let filteredPicks: EnrichedPick[] = [];
  if (allEnriched.length) {
    filteredPicks = allEnriched.filter(
      (p) => p.dfs_line != null && Number.isFinite(Number(p.dfs_line)),
    );
    filteredPicks = filterByPlatform(
      filteredPicks,
      activePlatform === "ALL" ? null : activePlatform,
    );
    filteredPicks = filterPlayerRowsByStat(filteredPicks, activeStat);
    filteredPicks = filterPlayerRows(filteredPicks, search);
  }
  const players = allEnriched.length
    ? sortPlayerGroups(aggregateEnrichedByPlayer(filteredPicks), groupSort)
    : [];
  const uniquePlayerCount = players.length;

  let panel: ReactNode;
  if (isLoading) {
    panel = (
      <>
        <LoadingMessage>Loading player data…</LoadingMessage>
        <PlayersListSkeleton count={5} />
      </>
    );
  } else if (isError) {
    panel = (
      <ErrorPanel
        message={error instanceof Error ? error.message : "Failed to load player data."}
        onRetry={() => refetch()}
      />
    );
  } else if (!allEnriched.length) {
    panel = (
      <p className="load-msg">
        No player props for this slate yet. Check back when games are posted, or try another
        league.
      </p>
    );
  } else if (!players.length) {
    panel = <div className="players-empty-state">No players match your filters.</div>;
  } else {
    panel = (
      <div className="players-grouped">
        <div className="players-grouped-header">
          <div className="players-grouped-count">
            <span className="dim">{filteredPicks.length} props</span>
            {dataSource && (
              <span className={`data-source-badge data-source-badge--${dataSource}`}>
                {dataSource === "api" ? "Live API" : "Static JSON"}
                {` · ${leagueBadgeLabel(league)}`}
                {gameDate ? ` · ${gameDate}` : ""}
              </span>
            )}
            {isFetching && !isLoading && (
              <span className="data-source-badge data-source-badge--refreshing">Updating…</span>
            )}
          </div>
          <div className="players-sort-bar" role="group" aria-label="Sort players">
            <span>Sort</span>
            {(
              [
                ["edge_desc", "Best edge"],
                ["props_desc", "Most props"],
                ["name_asc", "Name"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                data-grp-sort={key}
                aria-pressed={groupSort === key}
                onClick={() => setGroupSort(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {players.map((p) => (
          <PlayerBlock
            key={`${p.playerId ?? p.player}`}
            player={p}
            expanded={expandedPlayer === p.player}
            onToggle={() =>
              setExpandedPlayer((cur) => (cur === p.player ? null : p.player))
            }
            sortKey={sortKey}
            sortDir={sortDir}
            onSort={handleSort}
          />
        ))}
      </div>
    );
  }

  return (
    <section className="view-section" aria-labelledby="headingPlayers">
      <header className="view-section-head view-section-head--players">
        <h2 className="view-section-title" id="headingPlayers">
          All Players
        </h2>
      </header>
      <div className="players-toolbar">
        <div className="players-toolbar-row players-toolbar-row--filters">
          <Dropdown
            id="playersLeagueDropdown"
            value={league}
            options={LEAGUE_OPTIONS}
            onChange={onLeagueChange}
            classPrefix="league"
          />
          <FilterPills
            ariaLabel="Filter by stat"
            active={activeStat}
            onSelect={setActiveStat}
            items={[
              { value: "ALL", label: "All" },
              { value: "PTS", label: "Pts" },
              { value: "AST", label: "Ast" },
              { value: "REB", label: "Reb" },
            ]}
          />
        </div>
        <div className="players-toolbar-row players-toolbar-row--search">
          <div className="players-search-group">
            <label className="player-search-label" htmlFor="playerSearch">
              Search player
            </label>
            <input
              type="search"
              id="playerSearch"
              className="player-search"
              placeholder="Search by player name…"
              autoComplete="off"
              spellCheck={false}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              disabled={isLoading}
            />
          </div>
          <Dropdown
            id="playersBookDropdown"
            value={activePlatform}
            options={PLATFORM_OPTIONS}
            onChange={setActivePlatform}
            classPrefix="book"
            ariaLabel="Book"
          />
          <p className="players-count" aria-live="polite">
            Count:{" "}
            <b>{isLoading ? "—" : uniquePlayerCount}</b>
          </p>
        </div>
      </div>
      <div id="playersPanel">{panel}</div>
    </section>
  );
}
