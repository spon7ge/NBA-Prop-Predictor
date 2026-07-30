import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
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
});
