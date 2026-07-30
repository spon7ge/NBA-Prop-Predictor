import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { HomeChromeLayout } from "./HomeChromeLayout";

vi.mock("@/hooks/useWnbaScoreboard", () => ({
  useWnbaScoreboard: () => ({
    isLoading: false,
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

describe("HomeChromeLayout", () => {
  it("renders ticker games from scoreboard hook", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<HomeChromeLayout />}>
            <Route path="/" element={<div>home</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText("ATL")).toBeInTheDocument();
    expect(screen.getByText("DAL")).toBeInTheDocument();
  });
});
