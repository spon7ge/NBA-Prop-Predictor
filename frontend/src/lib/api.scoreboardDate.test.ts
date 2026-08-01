import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { fetchWnbaScoreboardByDate } from "./api";

describe("fetchWnbaScoreboardByDate", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ date: "2026-07-28", games: [], fetched_at: "" }),
      }),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it("calls /api/wnba/scoreboard?date=", async () => {
    await fetchWnbaScoreboardByDate("2026-07-28");
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/wnba/scoreboard?date=2026-07-28"),
      expect.any(Object),
    );
  });
});
