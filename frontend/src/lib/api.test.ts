import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const fetchMock = vi.fn();

const okResponse = {
  ok: true,
  status: 200,
  json: async () => ({ date: "2026-07-29", games: [], fetched_at: "now" }),
};

describe("fetchWnbaScoreboard", () => {
  beforeEach(() => {
    vi.resetModules();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("uses a relative path when VITE_API_BASE_URL is unset", async () => {
    vi.stubEnv("VITE_API_BASE_URL", undefined);
    fetchMock.mockResolvedValue(okResponse);

    const { fetchWnbaScoreboard } = await import("./api");
    await fetchWnbaScoreboard();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/wnba/scoreboard/today",
      expect.anything(),
    );
  });

  it("prefixes the configured API origin and strips a trailing slash", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "https://api.example.com/");
    fetchMock.mockResolvedValue(okResponse);

    const { fetchWnbaScoreboard } = await import("./api");
    await fetchWnbaScoreboard();

    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.com/api/wnba/scoreboard/today",
      expect.anything(),
    );
  });

  it("throws when the response is not ok", async () => {
    vi.stubEnv("VITE_API_BASE_URL", undefined);
    fetchMock.mockResolvedValue({ ok: false, status: 502 });

    const { fetchWnbaScoreboard } = await import("./api");
    await expect(fetchWnbaScoreboard()).rejects.toThrow(
      "Scoreboard request failed: 502",
    );
  });
});

describe("fetchGameDetail", () => {
  beforeEach(() => {
    vi.resetModules();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("hits /api/wnba/games/:id", async () => {
    vi.stubEnv("VITE_API_BASE_URL", undefined);
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ espn_event_id: "401857098", league: "wnba" }),
    });
    const { fetchGameDetail } = await import("./api");
    await fetchGameDetail("401857098");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/wnba/games/401857098",
      expect.objectContaining({ cache: "no-store" }),
    );
  });
});
