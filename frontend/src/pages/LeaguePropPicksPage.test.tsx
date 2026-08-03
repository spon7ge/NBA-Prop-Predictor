import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ApiWnbaPropLine } from "@/lib/api";
import { LeaguePropPicksPage } from "./LeaguePropPicksPage";

function prop(
  partial: Partial<ApiWnbaPropLine> &
    Pick<ApiWnbaPropLine, "player_name" | "stat" | "side" | "team_abbrev">,
): ApiWnbaPropLine {
  return {
    logo_url: null,
    market_type: "player_points",
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
    prizepicks: { line: 4.5, odds_american: 100 },
    underdog: null,
    betr: null,
    novig: null,
    sleeper: null,
    betrivers: null,
    ...partial,
  };
}

const mockProps: ApiWnbaPropLine[] = [
  prop({
    player_name: "Rhyne Howard",
    team_abbrev: "ATL",
    stat: "Assists",
    side: "over",
    market_type: "prizepicks:Assists",
  }),
  prop({
    player_name: "Jewell Loyd",
    team_abbrev: "SEA",
    stat: "Points",
    side: "over",
    market_type: "prizepicks:Points",
  }),
];

vi.mock("@/hooks/useWnbaProps", () => ({
  useWnbaProps: () => ({
    data: { props: mockProps, error: null },
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

  it("filters the table when a team is selected", async () => {
    const user = userEvent.setup();
    renderPage();

    expect(await screen.findByText("Rhyne Howard")).toBeInTheDocument();
    expect(screen.getByText("Jewell Loyd")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Team" }));
    await user.click(screen.getByRole("option", { name: /ATL/i }));

    expect(screen.getByRole("button", { name: "Team (1)" })).toBeInTheDocument();
    expect(screen.getByText("Rhyne Howard")).toBeInTheDocument();
    expect(screen.queryByText("Jewell Loyd")).toBeNull();
  });
});
