import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { LiveTicker } from "./LiveTicker";
import type { TickerGame } from "./types";

const game: TickerGame = {
  id: "g1",
  league: "wnba",
  awayAbbrev: "ATL",
  homeAbbrev: "DAL",
  statusLabel: "Q3 7:13",
  status: "live",
  awayScore: 36,
  homeScore: 44,
};

describe("LiveTicker", () => {
  it("shows the empty copy when there are no games", () => {
    render(<LiveTicker games={[]} />);
    expect(screen.getByText("No live games")).toBeInTheDocument();
  });

  it("shows unavailable copy when the scoreboard never loaded", () => {
    render(<LiveTicker games={[]} isError />);
    expect(screen.getByText("Scoreboard unavailable")).toBeInTheDocument();
  });

  it("renders games instead of the error copy when data is present", () => {
    render(<LiveTicker games={[game]} isError />);
    expect(screen.queryByText("Scoreboard unavailable")).not.toBeInTheDocument();
    expect(screen.getByText("ATL")).toBeInTheDocument();
    expect(screen.getByText("Q3 7:13")).toBeInTheDocument();
  });
});
