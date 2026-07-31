import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useGameDetail } from "./useGameDetail";

const fetchMock = vi.fn();

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useGameDetail", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("enables polling for live games", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        espn_event_id: "1",
        league: "wnba",
        status: "live",
        status_label: "Q1 4:13",
        venue: null,
        away: { id: "a", abbrev: "GS", name: "GS", score: 10, color: "#553987", logo_url: null },
        home: { id: "b", abbrev: "PHX", name: "PHX", score: 9, color: "#E56020", logo_url: null },
        fg_made: 0,
        fg_attempted: 0,
        latest_play: null,
        shots: [],
        plays: [],
        fetched_at: "",
      }),
    });
    const { result } = renderHook(() => useGameDetail("1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.shouldPoll).toBe(true);
  });

  it("disables polling for final games", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        espn_event_id: "1",
        league: "wnba",
        status: "final",
        status_label: "Final",
        venue: null,
        away: { id: "a", abbrev: "GS", name: "GS", score: 80, color: "#553987", logo_url: null },
        home: { id: "b", abbrev: "PHX", name: "PHX", score: 75, color: "#E56020", logo_url: null },
        fg_made: 0,
        fg_attempted: 0,
        latest_play: null,
        shots: [],
        plays: [],
        fetched_at: "",
      }),
    });
    const { result } = renderHook(() => useGameDetail("1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.shouldPoll).toBe(false);
  });

  it("flags hasNeverLoaded on first failure", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 502 });
    const { result } = renderHook(() => useGameDetail("1"), { wrapper });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.hasNeverLoaded).toBe(true);
  });
});
