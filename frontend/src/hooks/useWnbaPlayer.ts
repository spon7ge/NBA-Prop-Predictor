import { useQuery } from "@tanstack/react-query";
import { fetchWnbaPlayer } from "@/lib/api";

export function useWnbaPlayer(playerId: string) {
  const query = useQuery({
    queryKey: ["wnba", "player", playerId],
    queryFn: () => fetchWnbaPlayer(playerId),
    enabled: Boolean(playerId),
  });

  return {
    ...query,
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
