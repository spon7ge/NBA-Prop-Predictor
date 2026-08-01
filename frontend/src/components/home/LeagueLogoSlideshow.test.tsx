import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { LeagueLogoSlideshow } from "./LeagueLogoSlideshow";

function mockMatchMedia(reducedMotion: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches:
      reducedMotion && query === "(prefers-reduced-motion: reduce)",
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

describe("LeagueLogoSlideshow", () => {
  beforeEach(() => {
    mockMatchMedia(false);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders NBA and WNBA logo images", () => {
    render(<LeagueLogoSlideshow />);
    expect(screen.getByAltText("NBA")).toBeInTheDocument();
    expect(screen.getByAltText("WNBA")).toBeInTheDocument();
  });

  it("shows only the first logo when prefers-reduced-motion is set", () => {
    mockMatchMedia(true);
    render(<LeagueLogoSlideshow />);
    expect(screen.getByAltText("NBA")).toBeInTheDocument();
    expect(screen.queryByAltText("WNBA")).not.toBeInTheDocument();
  });
});
