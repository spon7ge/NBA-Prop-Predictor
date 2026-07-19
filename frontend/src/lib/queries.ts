import { useQuery } from "@tanstack/react-query";
import { fetchEnrichedPicks as fetchStaticEnrichedPicks } from "@/lib/api";
import {
  fetchLiveProps,
  fetchLiveSlates,
  fetchPerformance,
  fetchPlayerProfile,
  isApiAvailable,
  todayIso,
} from "@/lib/backend";
import { BOOKS, SLATE_LEG_COUNTS } from "@/lib/constants";
import { enrichSlatesFromLiveProps } from "@/lib/enrichSlateFromLiveProps";
import { mapLivePropsToEnrichedPicks } from "@/lib/mapLiveProps";
import { filterSupportedPicks } from "@/lib/players";
import { queryKeys } from "@/lib/queryClient";
import {
  hasAnySlates,
  loadAllSlates,
  normalizeSlateArray,
} from "@/lib/slate";
import type { ApiLeague, ApiLeagueFilter, ApiLiveSlatesResponse } from "@/types/api";
import type { Book, EnrichedPick, FlatParlayRow, LegCount } from "@/types/slate";

export type DataSource = "api" | "static";

export type SlatesByBook = Record<Book, FlatParlayRow[]>;
export type SlatesState = Record<LegCount, SlatesByBook>;

interface EnrichedPicksResult {
  picks: EnrichedPick[];
  source: DataSource;
  gameDate: string | null;
  league: ApiLeagueFilter;
}

interface LiveSlatesResult {
  slates: SlatesState;
  source: DataSource;
  gameDate: string | null;
  league: ApiLeagueFilter;
  count: number;
  /** Pipeline write time for this slate run (ISO), if from API. */
  runAt: string | null;
}

function emptySlatesState(): SlatesState {
  const out = {} as SlatesState;
  for (const n of SLATE_LEG_COUNTS) {
    out[n] = {} as SlatesByBook;
    for (const book of BOOKS) {
      out[n][book] = [];
    }
  }
  return out;
}

function sortByEvDesc(arr: FlatParlayRow[]): FlatParlayRow[] {
  return arr.slice().sort((a, b) => Number(b.EV) - Number(a.EV));
}

/** Map API live-slates envelope → TopLegs SlatesState. */
export function mapLiveSlatesResponse(res: ApiLiveSlatesResponse): SlatesState {
  const out = emptySlatesState();
  for (const n of SLATE_LEG_COUNTS) {
    const byBook = res.slates?.[String(n)] ?? {};
    for (const book of BOOKS) {
      const raw = byBook[book] ?? [];
      out[n][book] = sortByEvDesc(normalizeSlateArray(raw));
    }
  }
  return out;
}

async function loadOneLeague(
  league: ApiLeague,
  date: string,
): Promise<{ picks: EnrichedPick[]; gameDate: string | null }> {
  const live = await fetchLiveProps(league, date);
  const picks = filterSupportedPicks(mapLivePropsToEnrichedPicks(live)).map(
    (p) => ({ ...p, league }),
  );
  return { picks, gameDate: live.game_date };
}

async function loadEnrichedPicks(
  league: ApiLeagueFilter,
  date: string,
): Promise<EnrichedPicksResult> {
  const apiUp = await isApiAvailable();
  if (apiUp) {
    try {
      if (league === "all") {
        const [nba, wnba] = await Promise.allSettled([
          loadOneLeague("nba", date),
          loadOneLeague("wnba", date),
        ]);
        const picks: EnrichedPick[] = [];
        let gameDate: string | null = null;
        if (nba.status === "fulfilled") {
          picks.push(...nba.value.picks);
          gameDate = nba.value.gameDate ?? gameDate;
        }
        if (wnba.status === "fulfilled") {
          picks.push(...wnba.value.picks);
          gameDate = wnba.value.gameDate ?? gameDate;
        }
        if (picks.length > 0) {
          return { picks, source: "api", gameDate, league };
        }
      } else {
        const one = await loadOneLeague(league, date);
        if (one.picks.length > 0) {
          return {
            picks: one.picks,
            source: "api",
            gameDate: one.gameDate,
            league,
          };
        }
      }
    } catch {
      /* fall through to static */
    }
  }

  const staticPicks = filterSupportedPicks(await fetchStaticEnrichedPicks());
  return { picks: staticPicks, source: "static", gameDate: null, league };
}

async function loadLiveSlatesForLeague(
  league: ApiLeague,
  date: string,
): Promise<LiveSlatesResult | null> {
  const [live, propsSettled] = await Promise.all([
    fetchLiveSlates(league, date),
    loadOneLeague(league, date).catch(() => null),
  ]);
  const withContext =
    propsSettled && propsSettled.picks.length > 0
      ? enrichSlatesFromLiveProps(live, propsSettled.picks)
      : live;
  const slates = mapLiveSlatesResponse(withContext);
  if (!hasAnySlates(slates)) return null;
  return {
    slates,
    source: "api",
    gameDate: live.game_date,
    league,
    count: live.count,
    runAt: live.run_at ?? null,
  };
}

function mergeSlatesState(a: SlatesState, b: SlatesState): SlatesState {
  const out = emptySlatesState();
  for (const n of SLATE_LEG_COUNTS) {
    for (const book of BOOKS) {
      out[n][book] = sortByEvDesc([...(a[n][book] ?? []), ...(b[n][book] ?? [])]);
    }
  }
  return out;
}

async function loadLiveSlates(
  league: ApiLeagueFilter,
  date: string,
): Promise<LiveSlatesResult> {
  const apiUp = await isApiAvailable();
  if (apiUp) {
    try {
      if (league === "all") {
        const [nba, wnba] = await Promise.allSettled([
          loadLiveSlatesForLeague("nba", date),
          loadLiveSlatesForLeague("wnba", date),
        ]);
        const parts: LiveSlatesResult[] = [];
        if (nba.status === "fulfilled" && nba.value) parts.push(nba.value);
        if (wnba.status === "fulfilled" && wnba.value) parts.push(wnba.value);
        if (parts.length > 0) {
          let slates = parts[0].slates;
          for (let i = 1; i < parts.length; i += 1) {
            slates = mergeSlatesState(slates, parts[i].slates);
          }
          const runAts = parts
            .map((p) => p.runAt)
            .filter((t): t is string => Boolean(t));
          const runAt =
            runAts.length > 0
              ? runAts.reduce((a, b) => (a > b ? a : b))
              : null;
          return {
            slates,
            source: "api",
            gameDate: parts[0].gameDate,
            league: "all",
            count: parts.reduce((s, p) => s + p.count, 0),
            runAt,
          };
        }
      } else {
        const one = await loadLiveSlatesForLeague(league, date);
        if (one) return one;
      }
    } catch {
      /* fall through to static */
    }
  }

  const staticSlates = await loadAllSlates();
  return {
    slates: staticSlates,
    source: "static",
    gameDate: null,
    league,
    count: 0,
    runAt: null,
  };
}

export function useEnrichedPicks(
  league: ApiLeagueFilter = "wnba",
  date?: string,
) {
  const slateDate = date ?? todayIso();
  return useQuery({
    queryKey: queryKeys.enrichedPicks(league, slateDate),
    queryFn: () => loadEnrichedPicks(league, slateDate),
  });
}

export function useLiveSlates(league: ApiLeagueFilter = "wnba", date?: string) {
  const slateDate = date ?? todayIso();
  return useQuery({
    queryKey: queryKeys.liveSlates(league, slateDate),
    queryFn: () => loadLiveSlates(league, slateDate),
  });
}

export function usePlayerProfile(playerId: number | undefined, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.player(playerId ?? 0),
    queryFn: () => fetchPlayerProfile(playerId!, 10),
    enabled: enabled && playerId != null && playerId > 0,
  });
}

export type ResultsLegsFilter = "all" | "singles" | "2" | "3" | "5" | "6";

export function usePerformance(
  league: ApiLeagueFilter = "wnba",
  days = 7,
  book: string | "all" = "all",
  legs: ResultsLegsFilter = "all",
) {
  const bookKey = book === "all" ? "all" : book;
  return useQuery({
    queryKey: queryKeys.performance(league, days, bookKey, legs),
    queryFn: async () => {
      const apiUp = await isApiAvailable();
      if (!apiUp) throw new Error("API unavailable");
      const lg: ApiLeague = league === "nba" ? "nba" : "wnba";
      return fetchPerformance(
        lg,
        days,
        book === "all" ? null : book,
        legs === "all" ? null : legs,
      );
    },
  });
}
