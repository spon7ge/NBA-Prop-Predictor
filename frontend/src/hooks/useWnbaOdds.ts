import { useQuery } from "@tanstack/react-query";
import { fetchWnbaOdds } from "@/lib/api";

const REFETCH_MS = 60_000;

export function useWnbaOdds() {
  return useQuery({
    queryKey: ["wnba", "odds", "today"],
    queryFn: fetchWnbaOdds,
    refetchInterval: REFETCH_MS,
  });
}
