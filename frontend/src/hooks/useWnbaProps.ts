import { useQuery } from "@tanstack/react-query";
import { fetchWnbaProps } from "@/lib/api";

const REFETCH_MS = 60_000;

export function useWnbaProps() {
  return useQuery({
    queryKey: ["wnba", "props", "today"],
    queryFn: fetchWnbaProps,
    refetchInterval: REFETCH_MS,
  });
}
