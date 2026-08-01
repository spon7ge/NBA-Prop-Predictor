import type { GameDetail, GameDetailBoxScorePlayer, GameDetailTeam } from "./types";
import { GameSection } from "./GameSection";

const STAT_COLS =
  "grid-cols-[minmax(8rem,1.4fr)_repeat(14,minmax(2.1rem,1fr))]";

function TeamBoxScore({
  team,
  players,
  columns,
}: {
  team: GameDetailTeam;
  players: GameDetailBoxScorePlayer[];
  columns: string[];
}) {
  return (
    <div className="overflow-x-auto">
      <div className="mb-3 flex items-baseline gap-2 text-sm">
        <span className="font-semibold" style={{ color: team.color }}>
          {team.abbrev}
        </span>
        <span className="font-medium text-white">{team.name}</span>
      </div>

      <div
        className={`grid ${STAT_COLS} gap-x-2 border-b border-white/[0.08] pb-2 text-[10px] tracking-wide text-white/40`}
      >
        <span>Player</span>
        {columns.map((column) => (
          <span key={column} className="text-right uppercase">
            {column}
          </span>
        ))}
      </div>

      <ul>
        {players.map((player) => (
          <li
            key={`${team.id}-${player.name}`}
            className={`grid ${STAT_COLS} gap-x-2 border-b border-white/[0.06] py-2.5 text-[12px]`}
          >
            <span className="truncate text-white">{player.name}</span>
            {player.didNotPlay ? (
              <>
                {columns.slice(0, -1).map((column) => (
                  <span key={column} className="text-right text-white/30" />
                ))}
                <span className="text-right text-white/45">DNP</span>
              </>
            ) : (
              player.values.map((value, index) => (
                <span
                  key={`${player.name}-${columns[index] ?? index}`}
                  className="text-right tabular-nums text-white/85"
                >
                  {value}
                </span>
              ))
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function BoxScore({ detail }: { detail: GameDetail }) {
  const boxScore = detail.boxScore;
  if (!boxScore) {
    return null;
  }
  if (boxScore.away.length === 0 && boxScore.home.length === 0) {
    return null;
  }

  return (
    <GameSection className="space-y-10">
      {boxScore.away.length > 0 ? (
        <TeamBoxScore
          team={detail.away}
          players={boxScore.away}
          columns={boxScore.columns}
        />
      ) : null}
      {boxScore.home.length > 0 ? (
        <TeamBoxScore
          team={detail.home}
          players={boxScore.home}
          columns={boxScore.columns}
        />
      ) : null}
    </GameSection>
  );
}
