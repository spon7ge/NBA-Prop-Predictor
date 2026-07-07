import { useEffect, useState, type ReactNode } from "react";
import type { EnrichedPick, PlayersGroupSort } from "@/types/slate";
import type { PlayersSortKey } from "@/lib/constants";
import { fetchEnrichedPicks } from "@/lib/api";
import {
  aggregateEnrichedByPlayer,
  filterByPlatform,
  filterByTier,
  filterPlayerRows,
  filterPlayerRowsByStat,
  filterSupportedPicks,
  sortPlayerGroups,
} from "@/lib/players";
import { PlayerBlock } from "@/components/PlayerBlock";

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

export function AllPlayersView() {
  const [allEnriched, setAllEnriched] = useState<EnrichedPick[]>([]);
  const [state, setState] = useState<"idle" | "loading" | "loaded">("idle");
  const [search, setSearch] = useState("");
  const [activeStat, setActiveStat] = useState<string | null>(null);
  const [activePlatform, setActivePlatform] = useState<string | null>(null);
  const [activeTier, setActiveTier] = useState<string | null>(null);
  const [expandedPlayer, setExpandedPlayer] = useState<string | null>(null);
  const [groupSort, setGroupSort] = useState<PlayersGroupSort>("edge_desc");
  const [sortKey, setSortKey] = useState<PlayersSortKey | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    if (state !== "idle") return;
    setState("loading");
    fetchEnrichedPicks()
      .then((picks) => {
        setAllEnriched(filterSupportedPicks(picks));
        setState("loaded");
      })
      .catch(() => setState("loaded"));
  }, [state]);

  function handleSort(key: PlayersSortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  let panel: ReactNode;
  if (state === "loading") {
    panel = <p className="load-msg">Loading player data…</p>;
  } else if (!allEnriched.length) {
    panel = (
      <p className="load-msg load-err">
        No player data found. Ensure <code>dfs_enriched_YYYYMMDD.json</code> exists under{" "}
        <code>data/props/enriched/</code>.
      </p>
    );
  } else {
    let picks = allEnriched.slice();
    picks = filterByPlatform(picks, activePlatform);
    picks = filterByTier(picks, activeTier);
    picks = filterPlayerRowsByStat(picks, activeStat);
    picks = filterPlayerRows(picks, search);
    const players = sortPlayerGroups(aggregateEnrichedByPlayer(picks), groupSort);

    if (!players.length) {
      panel = <div className="players-empty-state">No players match your filters.</div>;
    } else {
      panel = (
        <div className="players-grouped">
          <div className="players-grouped-header">
            <div className="players-grouped-count">
              <b>{players.length}</b> {players.length === 1 ? "player" : "players"} ·{" "}
              <span className="dim">{picks.length} props</span>
            </div>
            <div className="players-sort-bar" role="group" aria-label="Sort players">
              <span>Sort</span>
              {(
                [
                  ["edge_desc", "Best edge"],
                  ["props_desc", "Most props"],
                  ["verified_desc", "Most verified"],
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
              key={p.player}
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
  }

  return (
    <section className="view-section" aria-labelledby="headingPlayers">
      <header className="view-section-head view-section-head--players">
        <h2 className="view-section-title" id="headingPlayers">
          All Players
        </h2>
      </header>
      <div className="players-toolbar">
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
          />
        </div>
        <div className="stat-filter-divider" aria-hidden="true" />
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
        <div className="stat-filter-divider" aria-hidden="true" />
        <FilterPills
          id="platformFilter"
          ariaLabel="Filter by platform"
          active={activePlatform}
          onSelect={setActivePlatform}
          items={[
            { value: "ALL", label: "All Platforms" },
            { value: "PrizePicks", label: "PrizePicks" },
            { value: "Underdog", label: "Underdog" },
            { value: "DraftKings Pick6", label: "DraftKings" },
            { value: "Betr DFS", label: "Betr" },
          ]}
        />
        <div className="stat-filter-divider" aria-hidden="true" />
        <FilterPills
          id="tierFilter"
          ariaLabel="Filter by tier"
          active={activeTier}
          onSelect={setActiveTier}
          getClassName={(v) =>
            v === "sharp_verified"
              ? "stat-pill--tier-sharp"
              : v === "conflict"
                ? "stat-pill--tier-conflict"
                : v === "no_model"
                  ? "stat-pill--tier-nomodel"
                  : ""
          }
          items={[
            { value: "ALL", label: "All Tiers" },
            { value: "sharp_verified", label: "Verified" },
            { value: "conflict", label: "Conflict" },
            { value: "no_model", label: "No Model" },
          ]}
        />
      </div>
      <div id="playersPanel">{panel}</div>
    </section>
  );
}
