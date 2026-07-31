import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LeagueStandingsPage } from "./LeagueStandingsPage";

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/wnba/standings"]}>
        <LeagueStandingsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeagueStandingsPage", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders standings from API", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        season: 2026,
        conferences: [
          {
            key: "east",
            label: "Eastern Conference",
            teams: [
              {
                rank: 1,
                team_id: "5",
                abbrev: "IND",
                name: "Indiana Fever",
                logo_url: null,
                wins: 18,
                losses: 10,
                wl: "18-10",
                pct: ".643",
                gb: "-",
                home: "11-5",
                away: "7-5",
                l10: "8-2",
                diff: "+169",
                streak: "W4",
              },
            ],
          },
        ],
      }),
    });

    renderPage();
    expect(await screen.findByText("Eastern Conference")).toBeInTheDocument();
    expect(screen.getByText("Indiana Fever")).toBeInTheDocument();
    expect(screen.getByText("Data: ESPN")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Standings" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });
});
