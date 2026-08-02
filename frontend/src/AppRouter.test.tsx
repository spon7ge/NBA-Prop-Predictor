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
    window.matchMedia = vi.fn().mockImplementation(() => ({
      matches: false,
      media: "",
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
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

  it("renders WNBA prop picks at /wnba/prop_picks", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        as_of: "now",
        sportsbooks: [
          "fanduel",
          "draftkings",
          "caesars",
          "betmgm",
          "pinnacle",
          "bet365",
          "prizepicks",
          "underdog",
          "betr",
          "novig",
          "sleeper",
          "pick6",
        ],
        props: [
          {
            player_name: "Rhyne Howard",
            team_abbrev: "ATL",
            logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
            stat: "Assists",
            market_type: "player_assists",
            side: "over",
            model_prediction: null,
            over_under_pct: null,
            ev: null,
            game_date: "2026-07-31",
            commence_time: "2026-07-31T23:30:00Z",
            fanduel: { line: 3.5, odds_american: -114 },
            draftkings: { line: 3.5, odds_american: -120 },
            caesars: null,
            betmgm: null,
            pinnacle: { line: 3.5, odds_american: -108 },
            bet365: null,
            prizepicks: { line: 3.5, odds_american: null },
            underdog: null,
            betr: null,
            novig: null,
            sleeper: null,
            pick6: null,
          },
        ],
      }),
    });
    renderWithProviders(["/wnba/prop_picks"]);
    expect(
      await screen.findByRole("heading", { name: "Prop Picks" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Rhyne Howard")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Prop Picks" })).toHaveAttribute(
      "aria-current",
      "page",
    );
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
              logo_url: null,
            },
            home: {
              id: "21",
              abbrev: "PHX",
              name: "Phoenix Mercury",
              score: 9,
              color: "#E56020",
              logo_url: null,
            },
            fg_made: 1,
            fg_attempted: 2,
            latest_play: null,
            shots: [],
            plays: [],
            win_probability: null,
            matchup_prediction: null,
            projected_starters: null,
            season_leaders: null,
            injuries: null,
            box_score: null,
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

  it("renders WNBA standings at /wnba/standings", async () => {
    fetchMock.mockImplementation(async (input: RequestInfo) => {
      const url = String(input);
      if (url.includes("/api/wnba/standings")) {
        return {
          ok: true,
          json: async () => ({
            season: 2026,
            conferences: [],
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
      };
    });
    renderWithProviders(["/wnba/standings"]);
    expect(
      await screen.findByText(/2026 regular season/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Data: ESPN")).toBeInTheDocument();
  });

  it("renders WNBA leaders at /wnba/leaders", async () => {
    fetchMock.mockImplementation(async (input: RequestInfo) => {
      const url = String(input);
      if (url.includes("/api/wnba/leaders")) {
        return {
          ok: true,
          json: async () => ({
            season: 2026,
            pace: "per_game",
            categories: [],
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
      };
    });
    renderWithProviders(["/wnba/leaders"]);
    expect(
      await screen.findByText(/2026 season · per game/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Data: stats.wnba.com")).toBeInTheDocument();
  });

  it("renders WNBA futures at /wnba/futures", async () => {
    fetchMock.mockImplementation(async (input: RequestInfo) => {
      const url = String(input);
      if (url.includes("/api/wnba/futures")) {
        return {
          ok: true,
          json: async () => ({
            season: 2026,
            as_of: "2026-08-01T00:00:00Z",
            markets: [
              {
                id: "8146",
                name: "WNBA - Winner",
                display_name: "Finals Winner",
                provider: "ESPN BET",
                entries: [
                  {
                    team_id: "8",
                    abbrev: "NYL",
                    name: "New York Liberty",
                    logo_url: null,
                    odds_american: "+250",
                  },
                ],
              },
            ],
            error: null,
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
      };
    });
    renderWithProviders(["/wnba/futures"]);
    expect(await screen.findByText("Finals Winner")).toBeInTheDocument();
    expect(await screen.findByText("New York Liberty")).toBeInTheDocument();
    expect(screen.getByText("+250")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Futures" })).toHaveAttribute(
      "aria-current",
      "page",
    );
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
              logo_url: null,
            },
            home: {
              id: "21",
              abbrev: "PHX",
              name: "Phoenix Mercury",
              score: 9,
              color: "#E56020",
              logo_url: null,
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
            matchup_prediction: null,
            projected_starters: null,
            season_leaders: null,
            injuries: null,
            box_score: null,
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
