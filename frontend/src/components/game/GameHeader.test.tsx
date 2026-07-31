import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { GameHeader } from "./GameHeader";
import { detail } from "./testFixtures";
import type { GameDetail } from "./types";

function renderHeader(overrides: Partial<GameDetail> = {}) {
  return render(
    <MemoryRouter>
      <GameHeader detail={{ ...detail, ...overrides }} />
    </MemoryRouter>,
  );
}

describe("GameHeader", () => {
  it("renders team names", () => {
    renderHeader();
    expect(screen.getByText("Golden State Valkyries")).toBeInTheDocument();
    expect(screen.getByText("Phoenix Mercury")).toBeInTheDocument();
  });

  it("renders team scores", () => {
    renderHeader();
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("9")).toBeInTheDocument();
  });

  it("renders status and venue inside the header card", () => {
    renderHeader();
    expect(screen.getAllByText("4:13 - 1st").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Mortgage Matchup Center")).toBeInTheDocument();
  });

  it("renders the status label in the top bar", () => {
    renderHeader();
    expect(screen.getAllByText("4:13 - 1st").length).toBeGreaterThanOrEqual(1);
  });

  it("links Back to the home page", () => {
    renderHeader();
    expect(screen.getByRole("link", { name: /back/i })).toHaveAttribute(
      "href",
      "/",
    );
  });

  it("shows a dash for scores that are not available yet", () => {
    renderHeader({
      away: { ...detail.away, score: null },
      home: { ...detail.home, score: null },
    });
    expect(screen.getAllByText("–")).toHaveLength(2);
  });

  it("omits the venue separator when venue is unknown", () => {
    renderHeader({ venue: null });
    expect(screen.queryByText("Mortgage Matchup Center")).not.toBeInTheDocument();
    expect(screen.getAllByText("4:13 - 1st").length).toBeGreaterThanOrEqual(1);
  });

  it("renders team logos when logoUrl is set", () => {
    renderHeader({
      away: {
        ...detail.away,
        logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
      },
      home: {
        ...detail.home,
        logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
      },
    });
    const images = screen.getAllByRole("presentation");
    expect(images).toHaveLength(2);
    expect(images[0]).toHaveAttribute(
      "src",
      "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
    );
    expect(images[1]).toHaveAttribute(
      "src",
      "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
    );
  });

  it("omits logo images when logoUrl is null", () => {
    renderHeader({
      away: { ...detail.away, logoUrl: null },
      home: { ...detail.home, logoUrl: null },
    });
    expect(screen.queryByRole("presentation")).not.toBeInTheDocument();
  });
});
