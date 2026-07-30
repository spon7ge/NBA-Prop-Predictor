import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { MatchupGameCard } from "./MatchupGameCard";
import type { MatchupGame } from "./types";

const liveGame: MatchupGame = {
  id: "1",
  espnEventId: "401857098",
  league: "wnba",
  status: "live",
  statusLabel: "3:31 - 4th",
  venue: "Mortgage Matchup Center",
  venueCity: "Phoenix",
  away: {
    abbrev: "GS",
    name: "Golden State Valkyries",
    score: 77,
    record: "19-8",
  },
  home: {
    abbrev: "PHX",
    name: "Phoenix Mercury",
    score: 78,
    record: "10-18",
  },
};

function renderCard(game: MatchupGame) {
  return render(
    <MemoryRouter>
      <MatchupGameCard game={game} />
    </MemoryRouter>,
  );
}

describe("MatchupGameCard", () => {
  it("links to game detail when espnEventId is set", () => {
    renderCard(liveGame);
    expect(
      screen.getByRole("link", { name: /golden state valkyries/i }),
    ).toHaveAttribute("href", "/games/401857098");
  });

  it("shows venue · city, status, records, and scores", () => {
    renderCard(liveGame);
    expect(screen.getByText("3:31 - 4th")).toBeInTheDocument();
    expect(
      screen.getByText("Mortgage Matchup Center · Phoenix"),
    ).toBeInTheDocument();
    expect(screen.getByText("19-8")).toBeInTheDocument();
    expect(screen.getByText("10-18")).toBeInTheDocument();
    expect(screen.getByText("77")).toBeInTheDocument();
    expect(screen.getByText("78")).toBeInTheDocument();
  });

  it("omits venue line and records when absent", () => {
    renderCard({
      ...liveGame,
      espnEventId: null,
      venue: null,
      venueCity: null,
      away: { ...liveGame.away, record: null },
      home: { ...liveGame.home, record: null },
    });
    expect(screen.queryByText(/Mortgage/)).not.toBeInTheDocument();
    expect(screen.queryByText("19-8")).not.toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("hides score badges for scheduled games only", () => {
    renderCard({
      ...liveGame,
      status: "scheduled",
      statusLabel: "8:00 PM ET",
      away: { ...liveGame.away, score: null },
      home: { ...liveGame.home, score: null },
    });

    expect(screen.queryByText("77")).not.toBeInTheDocument();
    expect(screen.queryByText("78")).not.toBeInTheDocument();
    expect(screen.queryByText("Golden State Valkyries")).toBeInTheDocument();
    expect(screen.queryByText("Phoenix Mercury")).toBeInTheDocument();
  });
});
