import { useQuery } from "@tanstack/react-query";
import { fetchWnbaScoreboard } from "@/lib/api";
import {
  mapToLiveGames,
  mapToTickerGames,
  shouldPollScoreboard,
} from "@/components/home/mapScoreboard";

const REFETCH_MS = 18_000;

export function useWnbaScoreboard() {
  const query = useQuery({
    queryKey: ["wnba", "scoreboard", "today"],
    queryFn: fetchWnbaScoreboard,
    refetchInterval: (q) =>
      shouldPollScoreboard(q.state.data?.games) ? REFETCH_MS : false,
  });

  const games = query.data?.games ?? [];
  return {
    ...query,
    games,
    tickerGames: mapToTickerGames(games),
    liveGames: mapToLiveGames(games),
    shouldPoll: shouldPollScoreboard(query.data?.games),
    // Errors after a successful load keep showing the last good scoreboard, so
    // only a never-loaded query surfaces an error state to the UI.
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
