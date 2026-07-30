import type { TickerGame } from "./types";

type LiveTickerProps = {
  games?: TickerGame[];
  /** Set only when the scoreboard has never loaded, so good data is never replaced. */
  isError?: boolean;
};

export function LiveTicker({ games = [], isError = false }: LiveTickerProps) {
  return (
    <div className="border-b border-white/10 bg-[#0a0a0a]">
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
          <ul className="flex min-w-0 flex-1 items-center gap-6 overflow-x-auto text-xs whitespace-nowrap">
            {games.map((game) => (
              <li key={game.id} className="flex items-center gap-2 text-white/70">
                <span className="font-medium text-sky-400">{game.awayAbbrev}</span>
                <span className="text-white/30">@</span>
                <span className="font-medium text-rose-400">{game.homeAbbrev}</span>
                <span className="text-white/40">{game.statusLabel}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
