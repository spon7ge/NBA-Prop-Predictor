import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { LiveNowSection } from "./LiveNowSection";
import type { LiveGame } from "./types";

describe("LiveNowSection", () => {
  it("renders empty-state copy and skeleton placeholders", () => {
    const { container } = render(<LiveNowSection />);
    expect(screen.getByText("0 games in progress")).toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(3);
  });

  it("renders a live game card when games are provided", () => {
    const game: LiveGame = {
      id: "g1",
      league: "nba",
      statusLabel: "Q2 4:12",
      status: "live",
      away: { abbrev: "BOS", name: "Boston Celtics", score: 55 },
      home: { abbrev: "NYK", name: "New York Knicks", score: 52 },
    };
    render(<LiveNowSection games={[game]} />);
    expect(screen.getByText("1 game in progress")).toBeInTheDocument();
    expect(screen.getByText("BOS")).toBeInTheDocument();
    expect(screen.getByText("NYK")).toBeInTheDocument();
    expect(screen.getByText("55")).toBeInTheDocument();
  });
});
