import type { TickerGame } from "./types";

type LiveTickerProps = {
  games?: TickerGame[];
  /** Set only when the scoreboard has never loaded, so good data is never replaced. */
  isError?: boolean;
};

function isScheduledFormat(game: TickerGame): boolean {
  return (
    game.status === "scheduled" ||
    (game.awayScore === null && game.homeScore === null)
  );
}

function TickerItem({ game }: { game: TickerGame }) {
  const scheduled = isScheduledFormat(game);

  return (
    <li className="flex items-center gap-2 border-l border-white/10 px-5 font-mono text-xs text-white/70 first:border-l-0">
      <span className="font-medium text-sky-400">{game.awayAbbrev}</span>
      {scheduled ? (
        <>
          <span className="text-white/30">@</span>
          <span className="font-medium text-rose-400">{game.homeAbbrev}</span>
        </>
      ) : (
        <>
          {game.awayScore !== null ? (
            <span className="text-white/80">{game.awayScore}</span>
          ) : null}
          <span className="text-white/30">—</span>
          <span className="font-medium text-rose-400">{game.homeAbbrev}</span>
          {game.homeScore !== null ? (
            <span className="text-white/80">{game.homeScore}</span>
          ) : null}
        </>
      )}
      <span className="text-white/40">{game.statusLabel}</span>
    </li>
  );
}

function TickerGameList({
  games,
  keyPrefix,
}: {
  games: TickerGame[];
  keyPrefix: string;
}) {
  return (
    <ul className="flex shrink-0 items-center whitespace-nowrap">
      {games.map((game) => (
        <TickerItem key={`${keyPrefix}-${game.id}`} game={game} />
      ))}
    </ul>
  );
}

export function LiveTicker({ games = [], isError = false }: LiveTickerProps) {
  return (
    <div className="ticker-marquee border-b border-white/10 bg-[#0a0a0a]">
      <div className="mx-auto flex max-w-6xl items-center gap-4 overflow-hidden px-4 py-2 sm:px-6">
        <div className="flex shrink-0 items-center gap-2">
          <span
            className="size-1.5 animate-pulse rounded-full bg-red-500"
            aria-hidden
          />
          <span className="text-[10px] font-semibold tracking-widest text-red-400 uppercase">
            Live
          </span>
        </div>

        {games.length === 0 ? (
          <p className="truncate text-xs text-white/40">
            {isError ? "Scoreboard unavailable" : "No live games"}
          </p>
        ) : (
          <div className="ticker-marquee-viewport min-w-0 flex-1 overflow-hidden">
            <div className="ticker-marquee-track flex w-max items-center">
              <TickerGameList games={games} keyPrefix="a" />
              <div className="ticker-marquee-duplicate" aria-hidden="true">
                <TickerGameList games={games} keyPrefix="b" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
