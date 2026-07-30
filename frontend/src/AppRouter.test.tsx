import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppRouter } from "@/AppRouter";

function renderWithProviders(initialEntries: string[]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={initialEntries}>
        <AppRouter />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AppRouter", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders home at /", () => {
    renderWithProviders(["/"]);
    expect(
      screen.getByRole("heading", { name: /hoopvista/i }),
    ).toBeInTheDocument();
  });

  it("renders about at /about", () => {
    renderWithProviders(["/about"]);
    expect(
      screen.getByRole("heading", { name: /about hoopvista/i }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("main")).toHaveLength(1);
    expect(screen.getByText("No live games")).toBeInTheDocument();
  });

  it("renders not found for unknown paths", () => {
    renderWithProviders(["/slate"]);
    expect(
      screen.getByRole("heading", { name: /page not found/i }),
    ).toBeInTheDocument();
  });

  it("renders WNBA matchups hub at /wnba/matchups", async () => {
    renderWithProviders(["/wnba/matchups"]);
    expect(
      await screen.findByRole("heading", { name: /women.?s basketball/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Matchups" }),
    ).toBeInTheDocument();
  });

  it("renders NBA coming-soon hub at /nba/matchups", async () => {
    renderWithProviders(["/nba/matchups"]);
    expect(
      await screen.findByRole("heading", { name: /men.?s basketball/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/coming soon/i)).toBeInTheDocument();
  });

  it("renders not found for unknown league matchups", () => {
    renderWithProviders(["/mlb/matchups"]);
    expect(
      screen.getByRole("heading", { name: /page not found/i }),
    ).toBeInTheDocument();
  });

  it("renders game detail at /games/:espnEventId", async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (String(url).includes("/api/wnba/games/")) {
        return {
          ok: true,
          json: async () => ({
            espn_event_id: "401857098",
            league: "wnba",
            status: "live",
            status_label: "4:13 - 1st",
            venue: "Mortgage Matchup Center",
            away: {
              id: "129153",
              abbrev: "GS",
              name: "Golden State Valkyries",
              score: 10,
              color: "#553987",
            },
            home: {
              id: "21",
              abbrev: "PHX",
              name: "Phoenix Mercury",
              score: 9,
              color: "#E56020",
            },
            fg_made: 1,
            fg_attempted: 2,
            latest_play: null,
            shots: [],
            plays: [],
            win_probability: null,
            fetched_at: "",
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
      };
    });
    renderWithProviders(["/games/401857098"]);
    expect(
      await screen.findByText(/Golden State Valkyries/i),
    ).toBeInTheDocument();
    expect(screen.getByText("No live games")).toBeInTheDocument(); // chrome ticker still present
  });

  it("renders win probability beneath shot chart and play-by-play", async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (String(url).includes("/api/wnba/games/")) {
        return {
          ok: true,
          json: async () => ({
            espn_event_id: "401857098",
            league: "wnba",
            status: "live",
            status_label: "4:13 - 1st",
            venue: "Mortgage Matchup Center",
            away: {
              id: "129153",
              abbrev: "GS",
              name: "Golden State Valkyries",
              score: 10,
              color: "#553987",
            },
            home: {
              id: "21",
              abbrev: "PHX",
              name: "Phoenix Mercury",
              score: 9,
              color: "#E56020",
            },
            fg_made: 1,
            fg_attempted: 2,
            latest_play: null,
            shots: [],
            plays: [],
            win_probability: {
              summary: "Above the midline favors PHX",
              timeline: [
                {
                  id: "wp-1",
                  period: 1,
                  clock: "4:29",
                  away_score: 10,
                  home_score: 8,
                  away_win_pct: 46,
                  home_win_pct: 54,
                  team_id: "21",
                },
              ],
              team_stats: [
                {
                  key: "field_goal_pct",
                  label: "Field goal %",
                  away_value: 41,
                  home_value: 49,
                },
              ],
            },
            fetched_at: "",
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
      };
    });

    renderWithProviders(["/games/401857098"]);

    expect(await screen.findByText("Shot chart")).toBeInTheDocument();
    expect(await screen.findByText("Play-by-play")).toBeInTheDocument();
    expect(await screen.findByText("Win probability")).toBeInTheDocument();
    expect(screen.getByText("Field goal %")).toBeInTheDocument();
  });
});
