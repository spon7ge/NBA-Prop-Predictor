import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWnbaLeaders } from "./useWnbaLeaders";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useWnbaLeaders", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads leaders payload", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        season: 2026,
        pace: "per_game",
        categories: [
          {
            key: "points",
            label: "Points",
            stat: "PTS",
            leaders: [
              {
                rank: 1,
                player_id: "1",
                name: "A'ja Wilson",
                team_abbrev: "LVA",
                gp: 25,
                value: "26.2",
              },
            ],
          },
        ],
      }),
    });

    const { result } = renderHook(() => useWnbaLeaders(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.categories[0].leaders[0].name).toBe(
      "A'ja Wilson",
    );
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/wnba/leaders"),
      expect.any(Object),
    );
  });

  it("sets hasNeverLoaded on cold error", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 502 });
    const { result } = renderHook(() => useWnbaLeaders(), { wrapper });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.hasNeverLoaded).toBe(true);
  });
});
