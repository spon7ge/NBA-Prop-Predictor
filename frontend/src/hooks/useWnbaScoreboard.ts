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
  };
}
