import { useState } from "react";
import type { ApiWnbaPlayerGame } from "@/lib/api";

type PlayerRecentGamesProps = {
  games: ApiWnbaPlayerGame[];
};

const DEFAULT_VISIBLE = 5;

const COLUMNS = [
  { key: "game_date", label: "Date" },
  { key: "matchup", label: "Matchup" },
  { key: "min", label: "MIN" },
  { key: "pts", label: "PTS" },
  { key: "fg", label: "FG" },
  { key: "three_pt", label: "3PT" },
  { key: "ft", label: "FT" },
  { key: "reb", label: "REB" },
  { key: "ast", label: "AST" },
  { key: "to", label: "TO" },
  { key: "stl", label: "STL" },
  { key: "blk", label: "BLK" },
] as const;

export function PlayerRecentGames({ games }: PlayerRecentGamesProps) {
  const [expanded, setExpanded] = useState(false);
  const canToggle = games.length > DEFAULT_VISIBLE;
  const visibleGames =
    expanded || !canToggle ? games : games.slice(0, DEFAULT_VISIBLE);

  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <h3 className="mb-3 text-base font-semibold tracking-tight text-white">
        Recent games
      </h3>

      {games.length === 0 ? (
        <p className="text-sm text-white/40">No games yet</p>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-xs">
              <thead>
                <tr className="text-[10px] tracking-wide text-white/35 uppercase">
                  {COLUMNS.map((col) => (
                    <th
                      key={col.key}
                      className={`pb-2 font-medium ${
                        col.key === "matchup" || col.key === "game_date"
                          ? "text-left"
                          : "text-right"
                      }`}
                    >
                      {col.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visibleGames.map((game) => (
                  <tr key={game.game_id} className="border-t border-white/5">
                    {COLUMNS.map((col) => (
                      <td
                        key={col.key}
                        className={`py-1.5 tabular-nums ${
                          col.key === "matchup" || col.key === "game_date"
                            ? "text-left text-white/80"
                            : "text-right text-white"
                        } ${col.key === "pts" ? "font-semibold" : ""}`}
                      >
                        {game[col.key]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {canToggle ? (
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="mt-3 rounded-md border border-white/10 px-2.5 py-0.5 text-xs text-white/55 hover:text-white"
            >
              {expanded ? "Show less" : "See more"}
            </button>
          ) : null}
        </>
      )}
    </section>
  );
}
