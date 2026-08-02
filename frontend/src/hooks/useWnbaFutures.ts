import { useQuery } from "@tanstack/react-query";
import { fetchWnbaFutures } from "@/lib/api";

export function useWnbaFutures() {
  const query = useQuery({
    queryKey: ["wnba", "futures"],
    queryFn: fetchWnbaFutures,
  });

  return {
    ...query,
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
