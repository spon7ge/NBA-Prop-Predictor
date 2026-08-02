import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWnbaPlayer } from "./useWnbaPlayer";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useWnbaPlayer", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads player payload", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        player_id: "1628932",
        name: "A'ja Wilson",
        position: "C",
        team_name: "Las Vegas Aces",
        team_abbrev: "LVA",
        headshot_url: null,
        season: 2026,
        averages: {
          pts: "26.2",
          reb: "10.1",
          ast: "2.5",
          fg_pct: "52.0",
          fg3_pct: "33.0",
        },
        games: [
          {
            game_id: "1",
            game_date: "2026-07-01",
            matchup: "LVA vs. NYL",
            min: "32",
            pts: "28",
            fg: "11-20",
            three_pt: "1-3",
            ft: "5-6",
            reb: "12",
            ast: "3",
            to: "2",
            stl: "1",
            blk: "2",
          },
        ],
        source_label: "stats.wnba.com",
      }),
    });

    const { result } = renderHook(() => useWnbaPlayer("1628932"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.name).toBe("A'ja Wilson");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/wnba/player/1628932"),
      expect.any(Object),
    );
  });

  it("does not fetch when playerId is empty", async () => {
    const { result } = renderHook(() => useWnbaPlayer(""), { wrapper });
    expect(result.current.fetchStatus).toBe("idle");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("sets hasNeverLoaded on cold error", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 502 });
    const { result } = renderHook(() => useWnbaPlayer("1628932"), { wrapper });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.hasNeverLoaded).toBe(true);
  });
});
