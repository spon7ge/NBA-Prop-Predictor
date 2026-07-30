import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { MatchupsPanel } from "./MatchupsPanel";
import type { MatchupGame } from "./types";

const live: MatchupGame = {
  id: "live-1",
  espnEventId: "1",
  league: "wnba",
  status: "live",
  statusLabel: "Q4 3:31",
  away: { abbrev: "GS", name: "Golden State Valkyries", score: 77 },
  home: { abbrev: "PHX", name: "Phoenix Mercury", score: 78 },
};

const finalGame: MatchupGame = {
  id: "final-1",
  espnEventId: "2",
  league: "wnba",
  status: "final",
  statusLabel: "Final",
  away: { abbrev: "ATL", name: "Atlanta Dream", score: 82 },
  home: { abbrev: "DAL", name: "Dallas Wings", score: 81 },
};

function renderPanel(games: MatchupGame[], props = {}) {
  return render(
    <MemoryRouter>
      <MatchupsPanel games={games} {...props} />
    </MemoryRouter>,
  );
}

describe("MatchupsPanel", () => {
  it("splits live and rest, shows count and disabled Today control", () => {
    renderPanel([live, finalGame]);
    expect(screen.getByRole("heading", { name: "Matchups" })).toBeInTheDocument();
    expect(
      screen.getByText(
        "2 games · open a card for box score, play-by-play & win probability",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("LIVE NOW")).toBeInTheDocument();
    expect(screen.getByText("REST OF THE SLATE")).toBeInTheDocument();
    expect(screen.getByText("Today")).toBeInTheDocument();
    const prev = screen.getByRole("button", { name: /previous day/i });
    const next = screen.getByRole("button", { name: /next day/i });
    expect(prev).toBeDisabled();
    expect(next).toBeDisabled();
  });

  it("hides LIVE NOW when no in-progress games", () => {
    renderPanel([finalGame]);
    expect(screen.queryByText("LIVE NOW")).not.toBeInTheDocument();
    expect(screen.getByText("REST OF THE SLATE")).toBeInTheDocument();
  });

  it("shows muted empty copy when no games and not loading", () => {
    renderPanel([]);
    expect(screen.getByText(/no games/i)).toBeInTheDocument();
  });

  it("announces an error when matchups cannot load", () => {
    renderPanel([], { isError: true });
    expect(screen.getByRole("status")).toHaveTextContent(
      "Unable to load matchups",
    );
  });
});
