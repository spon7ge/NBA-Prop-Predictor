import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
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

const linkedLiveGame: TickerGame = {
  ...liveGame,
  espnEventId: "401857098",
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
    expect(screen.getAllByText("ATL").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Q3 7:13").length).toBeGreaterThanOrEqual(1);
  });

  it("formats live games with scores and an em dash", () => {
    render(<LiveTicker games={[liveGame]} />);
    expect(screen.getAllByText("36").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("44").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText("@")).not.toBeInTheDocument();
  });

  it("formats scheduled games with @ and no scores", () => {
    render(<LiveTicker games={[scheduledGame]} />);
    expect(screen.getAllByText("@").length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText("—")).not.toBeInTheDocument();
  });

  it("duplicates the game list for the marquee track", () => {
    render(<LiveTicker games={[liveGame]} />);
    expect(screen.getAllByText("ATL")).toHaveLength(2);
  });

  it("marks the duplicate track as aria-hidden", () => {
    const { container } = render(<LiveTicker games={[liveGame]} />);
    const duplicate = container.querySelector(".ticker-marquee-duplicate");
    expect(duplicate?.getAttribute("aria-hidden")).toBe("true");
    expect(duplicate?.textContent).toContain("ATL");
  });

  it("does not render focusable links inside the aria-hidden duplicate track", () => {
    const { container } = render(
      <MemoryRouter>
        <LiveTicker games={[linkedLiveGame]} />
      </MemoryRouter>,
    );
    const duplicate = container.querySelector(".ticker-marquee-duplicate");
    expect(duplicate?.querySelector("a")).toBeNull();
    expect(duplicate?.textContent).toContain("ATL");
  });

  it("links to game detail when espnEventId is present", () => {
    render(
      <MemoryRouter>
        <LiveTicker games={[linkedLiveGame]} />
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: /ATL/i })).toHaveAttribute(
      "href",
      "/games/401857098",
    );
  });
});
