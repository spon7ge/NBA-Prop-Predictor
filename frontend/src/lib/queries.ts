import { useQuery } from "@tanstack/react-query";
import { fetchEnrichedPicks as fetchStaticEnrichedPicks } from "@/lib/api";
import { fetchGameSlate, fetchPlayerProfile, isApiAvailable, todayIso } from "@/lib/backend";
import { mapSlateToEnrichedPicks } from "@/lib/mapSlate";
import { filterSupportedPicks } from "@/lib/players";
import { queryKeys } from "@/lib/queryClient";
import type { EnrichedPick } from "@/types/slate";

export type DataSource = "api" | "static";

interface EnrichedPicksResult {
  picks: EnrichedPick[];
  source: DataSource;
  gameDate: string | null;
}

async function loadEnrichedPicks(date: string): Promise<EnrichedPicksResult> {
  const apiUp = await isApiAvailable();
  if (apiUp) {
    try {
      const slate = await fetchGameSlate(date);
      const picks = filterSupportedPicks(mapSlateToEnrichedPicks(slate));
      if (picks.length > 0) {
        return { picks, source: "api", gameDate: slate.game_date };
      }
    } catch {
      /* fall through to static */
    }
  }

  const staticPicks = filterSupportedPicks(await fetchStaticEnrichedPicks());
  return { picks: staticPicks, source: "static", gameDate: null };
}

export function useEnrichedPicks(date?: string) {
  const slateDate = date ?? todayIso();
  return useQuery({
    queryKey: queryKeys.enrichedPicks(slateDate),
    queryFn: () => loadEnrichedPicks(slateDate),
  });
}

export function usePlayerProfile(playerId: number | undefined, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.player(playerId ?? 0),
    queryFn: () => fetchPlayerProfile(playerId!, 10),
    enabled: enabled && playerId != null && playerId > 0,
  });
}
