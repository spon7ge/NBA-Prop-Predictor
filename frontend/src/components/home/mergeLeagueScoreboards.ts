import type { LiveGame, TickerGame } from "./types";

export type LeagueScoreboardPart = {
  tickerGames: TickerGame[];
  liveGames: LiveGame[];
  isLoading: boolean;
  hasNeverLoaded: boolean;
  shouldPoll?: boolean;
};

export function mergeLeagueScoreboards(parts: LeagueScoreboardPart[]): {
  tickerGames: TickerGame[];
  liveGames: LiveGame[];
  isLoading: boolean;
  hasNeverLoaded: boolean;
  shouldPoll: boolean;
} {
  const tickerGames = parts.flatMap((p) => p.tickerGames);
  const liveGames = parts.flatMap((p) => p.liveGames);
  const anyLoading = parts.some((p) => p.isLoading);

  return {
    tickerGames,
    liveGames,
    // Skeletons only while waiting and we have nothing live to show yet.
    isLoading: anyLoading && liveGames.length === 0,
    hasNeverLoaded:
      parts.length > 0 && parts.every((p) => p.hasNeverLoaded === true),
    shouldPoll: parts.some((p) => p.shouldPoll === true),
  };
}
