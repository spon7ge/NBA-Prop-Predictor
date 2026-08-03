import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useMlbScoreboard } from "./useMlbScoreboard";

const fetchMlbScoreboard = vi.fn();
const fetchMlbScoreboardByDate = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchMlbScoreboard: (...args: unknown[]) => fetchMlbScoreboard(...args),
  fetchMlbScoreboardByDate: (...args: unknown[]) =>
    fetchMlbScoreboardByDate(...args),
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const liveMlbGame = {
  id: "mlb-9",
  mlb_game_pk: "9",
  league: "mlb" as const,
  status: "live" as const,
  status_label: "Top 3rd",
  start_time_et: "2026-08-02T23:00:00Z",
  venue: "Yankee Stadium",
  venue_city: "New York",
  away: {
    abbrev: "BOS",
    name: "Boston Red Sox",
    score: 2,
    record: "50-40",
    logo_url: null,
  },
  home: {
    abbrev: "NYY",
    name: "New York Yankees",
    score: 3,
    record: "55-35",
    logo_url: null,
  },
};

describe("useMlbScoreboard", () => {
  beforeEach(() => {
    fetchMlbScoreboard.mockReset();
    fetchMlbScoreboardByDate.mockReset();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("uses the today scoreboard query key and maps ticker/live games", async () => {
    fetchMlbScoreboard.mockResolvedValue({
      date: "2026-08-02",
      fetched_at: "2026-08-02T12:00:00-04:00",
      games: [liveMlbGame],
    });

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const { result } = renderHook(() => useMlbScoreboard(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchMlbScoreboard).toHaveBeenCalled();
    expect(
      client
        .getQueryCache()
        .findAll({ queryKey: ["mlb", "scoreboard", "today"] }),
    ).toHaveLength(1);
    expect(result.current.tickerGames).toEqual([
      expect.objectContaining({
        id: "mlb-9",
        league: "mlb",
        mlbGamePk: "9",
        awayAbbrev: "BOS",
        homeAbbrev: "NYY",
        status: "live",
      }),
    ]);
    expect(result.current.liveGames).toEqual([
      expect.objectContaining({
        id: "mlb-9",
        league: "mlb",
        mlbGamePk: "9",
        status: "live",
      }),
    ]);
    expect(result.current.shouldPoll).toBe(true);
  });

  it("disables polling when all games are final", async () => {
    fetchMlbScoreboard.mockResolvedValue({
      date: "2026-08-02",
      fetched_at: "2026-08-02T12:00:00-04:00",
      games: [
        {
          ...liveMlbGame,
          status: "final",
          status_label: "Final",
        },
      ],
    });

    const { result } = renderHook(() => useMlbScoreboard(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.shouldPoll).toBe(false);
  });

  it("flags hasNeverLoaded when the first request fails", async () => {
    fetchMlbScoreboard.mockRejectedValue(new Error("MLB scoreboard failed"));

    const { result } = renderHook(() => useMlbScoreboard(), { wrapper });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.hasNeverLoaded).toBe(true);
    expect(result.current.liveGames).toEqual([]);
  });
});
