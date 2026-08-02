import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LeaguePlayerPage } from "./LeaguePlayerPage";

const PLAYER_PAYLOAD = {
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
};

function renderPage(playerId = "1628932") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/wnba/player/${playerId}`]}>
        <Routes>
          <Route path="/wnba/player/:playerId" element={<LeaguePlayerPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeaguePlayerPage", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders player name, recent games, and attribution", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => PLAYER_PAYLOAD,
    });

    renderPage();

    expect(await screen.findByText("A'ja Wilson")).toBeInTheDocument();
    expect(screen.getByText("Recent games")).toBeInTheDocument();
    expect(screen.getByText("Data: stats.wnba.com")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Leaders" })).toBeInTheDocument();
  });

  it("shows Player not found on 404", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 404 });

    renderPage("999");

    expect(await screen.findByText("Player not found")).toBeInTheDocument();
  });
});
