import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWnbaScoreboard } from "./useWnbaScoreboard";

const fetchMock = vi.fn();

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useWnbaScoreboard", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("disables refetch interval when all games are final", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        date: "2026-07-29",
        fetched_at: "2026-07-29T12:00:00-04:00",
        games: [
          {
            id: "1",
            league: "wnba",
            status: "final",
            status_label: "Final",
            start_time_et: "2026-07-29T23:00:00Z",
            away: { abbrev: "ATL", name: "Atlanta Dream", score: 80 },
            home: { abbrev: "DAL", name: "Dallas Wings", score: 75 },
          },
        ],
      }),
    });

    const { result } = renderHook(() => useWnbaScoreboard(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.shouldPoll).toBe(false);
  });

  it("enables polling when any game is live", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        date: "2026-07-29",
        fetched_at: "2026-07-29T12:00:00-04:00",
        games: [
          {
            id: "1",
            league: "wnba",
            status: "live",
            status_label: "Q3 7:13",
            start_time_et: "2026-07-29T23:00:00Z",
            away: { abbrev: "ATL", name: "Atlanta Dream", score: 36 },
            home: { abbrev: "DAL", name: "Dallas Wings", score: 44 },
          },
        ],
      }),
    });

    const { result } = renderHook(() => useWnbaScoreboard(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.shouldPoll).toBe(true);
  });
});
