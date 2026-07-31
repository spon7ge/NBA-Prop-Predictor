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

const finalGame: TickerGame = {
  id: "g3",
  league: "wnba",
  awayAbbrev: "CHI",
  homeAbbrev: "MIN",
  statusLabel: "Final",
  status: "final",
  awayScore: 78,
  homeScore: 82,
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

  it("hides scheduled and final games", () => {
    render(<LiveTicker games={[scheduledGame, liveGame]} />);
    expect(screen.queryByText("NYL")).not.toBeInTheDocument();
    expect(screen.queryByText("@")).not.toBeInTheDocument();
    expect(screen.getAllByText("ATL").length).toBeGreaterThanOrEqual(1);
  });

  it("shows TODAY rail and scheduled games when none are live", () => {
    render(<LiveTicker games={[scheduledGame]} />);
    expect(screen.getByText("Today")).toBeInTheDocument();
    expect(screen.queryByText("No live games")).not.toBeInTheDocument();
    expect(screen.getAllByText("NYL").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("7:00 PM ET").length).toBeGreaterThanOrEqual(1);
  });

  it("shows TODAY rail and finals when the slate is finished", () => {
    render(<LiveTicker games={[finalGame]} />);
    expect(screen.getByText("Today")).toBeInTheDocument();
    expect(screen.getAllByText("CHI").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("78").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Final").length).toBeGreaterThanOrEqual(1);
  });

  it("orders scheduled before finals in TODAY mode", () => {
    const { container } = render(
      <LiveTicker games={[finalGame, scheduledGame]} />,
    );
    const track = container.querySelector(".ticker-marquee-track");
    const text = track?.textContent ?? "";
    expect(text.indexOf("NYL")).toBeGreaterThanOrEqual(0);
    expect(text.indexOf("NYL")).toBeLessThan(text.indexOf("CHI"));
  });

  it("keeps LIVE rail and hides scheduled when any game is live", () => {
    render(<LiveTicker games={[scheduledGame, liveGame]} />);
    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.queryByText("Today")).not.toBeInTheDocument();
    expect(screen.queryByText("NYL")).not.toBeInTheDocument();
    expect(screen.getAllByText("ATL").length).toBeGreaterThanOrEqual(1);
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
