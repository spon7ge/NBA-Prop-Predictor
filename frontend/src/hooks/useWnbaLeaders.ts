import { useQuery } from "@tanstack/react-query";
import { fetchWnbaLeaders } from "@/lib/api";

export function useWnbaLeaders() {
  const query = useQuery({
    queryKey: ["wnba", "leaders"],
    queryFn: fetchWnbaLeaders,
  });

  return {
    ...query,
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
