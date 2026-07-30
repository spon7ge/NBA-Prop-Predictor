import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { LiveNowSection } from "./LiveNowSection";
import type { LiveGame } from "./types";

const liveGame: LiveGame = {
  id: "g1",
  league: "wnba",
  status: "live",
  statusLabel: "Q3 7:13",
  away: { abbrev: "ATL", name: "Atlanta Dream", score: 36 },
  home: { abbrev: "DAL", name: "Dallas Wings", score: 44 },
};

const linkedLiveGame: LiveGame = {
  ...liveGame,
  espnEventId: "401857098",
};

const finalGame: LiveGame = {
  id: "g2",
  league: "wnba",
  status: "final",
  statusLabel: "Final",
  away: { abbrev: "NYL", name: "New York Liberty", score: 90 },
  home: { abbrev: "LAS", name: "Los Angeles Sparks", score: 80 },
};

describe("LiveNowSection", () => {
  it("shows skeletons while loading with no games", () => {
    const { container } = render(<LiveNowSection isLoading games={[]} />);
    expect(screen.getByText("0 games in progress")).toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(3);
  });

  it("shows empty state without skeletons when loaded with zero games", () => {
    const { container } = render(<LiveNowSection isLoading={false} games={[]} />);
    expect(screen.getByText("0 games in progress")).toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(0);
  });

  it("counts only in-progress games in the subtitle", () => {
    render(<LiveNowSection games={[liveGame, finalGame]} />);
    expect(screen.getByText("1 game in progress")).toBeInTheDocument();
    expect(screen.getByText("ATL")).toBeInTheDocument();
    expect(screen.getByText("NYL")).toBeInTheDocument();
  });

  it("shows a muted error when the scoreboard never loaded", () => {
    const { container } = render(<LiveNowSection isError games={[]} />);
    expect(screen.getByText("Unable to load scoreboard")).toBeInTheDocument();
    expect(container.querySelectorAll("article")).toHaveLength(0);
  });

  it("keeps showing games when an error follows a successful load", () => {
    render(<LiveNowSection isError={false} games={[liveGame]} />);
    expect(screen.queryByText("Unable to load scoreboard")).not.toBeInTheDocument();
    expect(screen.getByText("ATL")).toBeInTheDocument();
  });

  it("prefers skeletons over the error message while still loading", () => {
    const { container } = render(<LiveNowSection isError isLoading games={[]} />);
    expect(screen.queryByText("Unable to load scoreboard")).not.toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(3);
  });

  it("links to game detail when espnEventId is present", () => {
    render(
      <MemoryRouter>
        <LiveNowSection games={[linkedLiveGame]} />
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: /Atlanta Dream/i })).toHaveAttribute(
      "href",
      "/games/401857098",
    );
  });
});
