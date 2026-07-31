import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LeagueLeadersPage } from "./LeagueLeadersPage";

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/wnba/leaders"]}>
        <LeagueLeadersPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeagueLeadersPage", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders leaders from API", async () => {
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

    renderPage();
    expect(await screen.findByText("Points")).toBeInTheDocument();
    expect(screen.getByText("A'ja Wilson")).toBeInTheDocument();
    expect(screen.getByText("Data: stats.wnba.com")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Leaders" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });
});
