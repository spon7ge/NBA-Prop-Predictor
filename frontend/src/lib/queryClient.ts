import { QueryClient } from "@tanstack/react-query";

export const STALE_TIME_MS = 5 * 60 * 1000;
export const GC_TIME_MS = 30 * 60 * 1000;

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: STALE_TIME_MS,
      gcTime: GC_TIME_MS,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export const queryKeys = {
  health: ["health"] as const,
  slate: (date: string) => ["slate", date] as const,
  player: (playerId: number) => ["player", playerId] as const,
  enrichedPicks: (league: string, date: string) =>
    ["enrichedPicks", league, date] as const,
  liveSlates: (league: string, date: string) =>
    ["liveSlates", league, date] as const,
  performance: (league: string, days: number, book: string, legs: string) =>
    ["performance", league, days, book, legs] as const,
};
