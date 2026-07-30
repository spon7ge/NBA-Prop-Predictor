import { useQuery } from "@tanstack/react-query";
import { fetchGameDetail } from "@/lib/api";

export function useGameDetail(espnEventId: string | undefined) {
  const query = useQuery({
    queryKey: ["wnba", "game", espnEventId],
    queryFn: () => fetchGameDetail(espnEventId!),
    enabled: Boolean(espnEventId),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status === "live" || status === "halftime" ? 18_000 : false;
    },
  });
  const status = query.data?.status;
  const shouldPoll = status === "live" || status === "halftime";
  return {
    ...query,
    shouldPoll,
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
