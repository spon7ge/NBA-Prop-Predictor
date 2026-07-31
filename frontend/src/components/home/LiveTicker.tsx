import { Link } from "react-router-dom";
import { isInProgressStatus } from "./mapScoreboard";
import type { TickerGame } from "./types";

type LiveTickerProps = {
  games?: TickerGame[];
  /** Set only when the scoreboard has never loaded, so good data is never replaced. */
  isError?: boolean;
};

function TickerItem({
  game,
  interactive = true,
}: {
  game: TickerGame;
  interactive?: boolean;
}) {
  const isScheduled = game.status === "scheduled";

  const content = isScheduled ? (
    <>
      <span className="font-medium text-sky-400">{game.awayAbbrev}</span>
      <span className="text-white/30">@</span>
      <span className="font-medium text-rose-400">{game.homeAbbrev}</span>
      <span className="text-white/40">{game.statusLabel}</span>
    </>
  ) : (
    <>
      <span className="font-medium text-sky-400">{game.awayAbbrev}</span>
      {game.awayScore !== null ? (
        <span className="text-white/80">{game.awayScore}</span>
      ) : null}
      <span className="text-white/30">—</span>
      <span className="font-medium text-rose-400">{game.homeAbbrev}</span>
      {game.homeScore !== null ? (
        <span className="text-white/80">{game.homeScore}</span>
      ) : null}
      <span className="text-white/40">{game.statusLabel}</span>
    </>
  );

  const itemClassName =
    "flex items-center gap-2 border-l border-white/10 px-5 font-mono text-xs text-white/70 first:border-l-0";

  if (game.espnEventId && interactive) {
    return (
      <li className={itemClassName}>
        <Link to={`/games/${game.espnEventId}`} className="flex items-center gap-2">
          {content}
        </Link>
      </li>
    );
  }

  return <li className={itemClassName}>{content}</li>;
}

function TickerGameList({
  games,
  keyPrefix,
  interactive = true,
}: {
  games: TickerGame[];
  keyPrefix: string;
  interactive?: boolean;
}) {
  return (
    <ul className="flex shrink-0 items-center whitespace-nowrap">
      {games.map((game) => (
        <TickerItem
          key={`${keyPrefix}-${game.id}`}
          game={game}
          interactive={interactive}
        />
      ))}
    </ul>
  );
}

export function LiveTicker({ games = [], isError = false }: LiveTickerProps) {
  const liveGames = games.filter((g) => isInProgressStatus(g.status));
  const scheduledGames = games.filter((g) => g.status === "scheduled");
  const finalGames = games.filter((g) => g.status === "final");

  const mode: "live" | "today" | "empty" =
    liveGames.length > 0
      ? "live"
      : scheduledGames.length > 0 || finalGames.length > 0
        ? "today"
        : "empty";

  const displayGames =
    mode === "live"
      ? liveGames
      : mode === "today"
        ? [...scheduledGames, ...finalGames]
        : [];

  const isToday = mode === "today";

  return (
    <div className="ticker-marquee border-b border-white/10 bg-[#0a0a0a]">
      <div className="mx-auto flex max-w-6xl items-center gap-4 overflow-hidden px-4 py-2 sm:px-6">
        <div className="flex shrink-0 items-center gap-2">
          <span
            className={
              isToday
                ? "size-1.5 rounded-full bg-white/40"
                : "size-1.5 animate-pulse rounded-full bg-red-500"
            }
            aria-hidden
          />
          <span
            className={
              isToday
                ? "text-[10px] font-semibold tracking-widest text-white/50 uppercase"
                : "text-[10px] font-semibold tracking-widest text-red-400 uppercase"
            }
          >
            {isToday ? "Today" : "Live"}
          </span>
        </div>

        {mode === "empty" ? (
          <p className="truncate text-xs text-white/40">
            {isError ? "Scoreboard unavailable" : "No live games"}
          </p>
        ) : (
          <div className="ticker-marquee-viewport min-w-0 flex-1 overflow-hidden">
            <div className="ticker-marquee-track flex w-max items-center">
              <TickerGameList games={displayGames} keyPrefix="a" />
              <div className="ticker-marquee-duplicate" aria-hidden="true">
                <TickerGameList
                  games={displayGames}
                  keyPrefix="b"
                  interactive={false}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
