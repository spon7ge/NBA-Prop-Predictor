import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { LiveTicker } from "./LiveTicker";
import type { TickerGame } from "./types";

const liveGame: TickerGame = {
  id: "g1",
  league: "wnba",
  awayAbbrev: "ATL",
  homeAbbrev: "DAL",
  statusLabel: "Q3 7:13",
  status: "live",
  awayScore: 36,
  homeScore: 44,
};

const scheduledGame: TickerGame = {
  id: "g2",
  league: "wnba",
  awayAbbrev: "NYL",
  homeAbbrev: "LVA",
  statusLabel: "7:00 PM ET",
  status: "scheduled",
  awayScore: null,
  homeScore: null,
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
    render(<LiveTicker games={[liveGame]} isError />);
    expect(screen.queryByText("Scoreboard unavailable")).not.toBeInTheDocument();
    expect(screen.getByText("ATL")).toBeInTheDocument();
    expect(screen.getByText("Q3 7:13")).toBeInTheDocument();
  });

  it("formats live games with scores and an em dash", () => {
    render(<LiveTicker games={[liveGame]} />);
    expect(screen.getByText("36")).toBeInTheDocument();
    expect(screen.getByText("44")).toBeInTheDocument();
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByText("@")).not.toBeInTheDocument();
  });

  it("formats scheduled games with @ and no scores", () => {
    render(<LiveTicker games={[scheduledGame]} />);
    expect(screen.getByText("@")).toBeInTheDocument();
    expect(screen.queryByText("—")).not.toBeInTheDocument();
  });
});
