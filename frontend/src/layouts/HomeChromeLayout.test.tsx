import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { HomeChromeLayout } from "./HomeChromeLayout";

vi.mock("@/hooks/useWnbaScoreboard", () => ({
  useWnbaScoreboard: () => ({
    isLoading: false,
    hasNeverLoaded: false,
    tickerGames: [
      {
        id: "1",
        league: "wnba",
        awayAbbrev: "ATL",
        homeAbbrev: "DAL",
        statusLabel: "Q3 7:13",
        status: "live",
        awayScore: 36,
        homeScore: 44,
      },
    ],
    liveGames: [],
  }),
}));

vi.mock("@/hooks/useMlbScoreboard", () => ({
  useMlbScoreboard: () => ({
    isLoading: false,
    hasNeverLoaded: false,
    tickerGames: [
      {
        id: "mlb-9",
        league: "mlb",
        mlbGamePk: "9",
        awayAbbrev: "BOS",
        homeAbbrev: "NYY",
        statusLabel: "Top 3rd",
        status: "live",
        awayScore: 2,
        homeScore: 3,
      },
    ],
    liveGames: [],
  }),
}));

describe("HomeChromeLayout", () => {
  it("renders ticker games from merged WNBA and MLB scoreboards", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<HomeChromeLayout />}>
            <Route path="/" element={<div>home</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getAllByText("ATL").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("DAL").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("BOS").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("NYY").length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getByText(/informational and entertainment purposes only/i),
    ).toBeInTheDocument();
  });
});
