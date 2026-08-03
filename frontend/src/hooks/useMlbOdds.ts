import { useQuery } from "@tanstack/react-query";
import { fetchMlbOdds } from "@/lib/api";

const REFETCH_MS = 60_000;

export function useMlbOdds() {
  return useQuery({
    queryKey: ["mlb", "odds", "today"],
    queryFn: fetchMlbOdds,
    refetchInterval: REFETCH_MS,
  });
}
