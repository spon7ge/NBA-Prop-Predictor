import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ApiWnbaPropLine } from "@/lib/api";
import { LeaguePropPicksPage } from "./LeaguePropPicksPage";

const mockProp: ApiWnbaPropLine = {
  player_name: "Rhyne Howard",
  team_abbrev: "ATL",
  logo_url: null,
  stat: "Assists",
  side: "over",
  market_type: "player_assists",
  game_date: null,
  commence_time: null,
  model_prediction: null,
  over_under_pct: null,
  ev: null,
  fanduel: null,
  draftkings: null,
  caesars: null,
  betmgm: null,
  pinnacle: null,
  bet365: null,
  prizepicks: { line: 4.5, odds_american: -110 },
  underdog: { line: 4.5, odds_american: -115 },
  betr: null,
  novig: null,
  sleeper: null,
  pick6: null,
};

vi.mock("@/hooks/useWnbaProps", () => ({
  useWnbaProps: () => ({
    data: { props: [mockProp], error: null },
    isLoading: false,
    isError: false,
    isFetched: true,
    dataUpdatedAt: Date.UTC(2026, 7, 2, 12, 0),
  }),
}));

vi.mock("@/hooks/useWnbaScoreboard", () => ({
  useWnbaScoreboard: () => ({
    games: [],
    data: { date: "2026-08-02", games: [], fetched_at: "" },
  }),
}));

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/wnba/prop-picks"]}>
        <LeaguePropPicksPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeaguePropPicksPage", () => {
  it("defaults book filter to PrizePicks and Underdog", async () => {
    renderPage();

    expect(await screen.findByText("Rhyne Howard")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Book (2)" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "PrizePicks" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Underdog" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "FanDuel" })).toBeNull();
  });
});
