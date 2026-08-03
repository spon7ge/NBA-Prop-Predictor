import { useQuery } from "@tanstack/react-query";
import { slateEtDate } from "@/components/league/matchupSlateDate";
import { fetchMlbScoreboard, fetchMlbScoreboardByDate } from "@/lib/api";
import {
  mapToLiveGames,
  mapToTickerGames,
  shouldPollScoreboard,
} from "@/components/home/mapScoreboard";

const REFETCH_MS = 18_000;

export function useMlbScoreboard(dateEt?: string) {
  const today = slateEtDate();
  const selected = dateEt ?? today;
  const isToday = selected === today;

  const query = useQuery({
    queryKey: isToday
      ? ["mlb", "scoreboard", "today"]
      : ["mlb", "scoreboard", selected],
    queryFn: () =>
      isToday ? fetchMlbScoreboard() : fetchMlbScoreboardByDate(selected),
    refetchInterval: (q) =>
      isToday && shouldPollScoreboard(q.state.data?.games)
        ? REFETCH_MS
        : false,
  });

  const games = query.data?.games ?? [];
  return {
    ...query,
    games,
    tickerGames: mapToTickerGames(games),
    liveGames: mapToLiveGames(games),
    shouldPoll: isToday && shouldPollScoreboard(query.data?.games),
    // Errors after a successful load keep showing the last good scoreboard, so
    // only a never-loaded query surfaces an error state to the UI.
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
