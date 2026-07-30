import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { formatSlateDateLabel, LeagueHero } from "./LeagueHero";

describe("LeagueHero", () => {
  it("renders WNBA hero copy", () => {
    render(<LeagueHero league="wnba" dateEt="2026-07-29" />);
    expect(screen.getByText("WNBA")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /women.?s basketball/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("WED, JUL 29")).toBeInTheDocument();
  });

  it("renders NBA hero copy", () => {
    render(<LeagueHero league="nba" />);
    expect(
      screen.getByRole("heading", { name: /men.?s basketball/i }),
    ).toBeInTheDocument();
  });

  it("formats slate dates in ET", () => {
    expect(formatSlateDateLabel("2026-07-29")).toBe("WED, JUL 29");
    expect(formatSlateDateLabel("2026-07-30")).toBe("THU, JUL 30");
  });
});
