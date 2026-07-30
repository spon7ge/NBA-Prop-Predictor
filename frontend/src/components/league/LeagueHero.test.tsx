import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LeagueHero } from "./LeagueHero";

describe("LeagueHero", () => {
  it("renders WNBA hero copy", () => {
    render(<LeagueHero league="wnba" />);
    expect(screen.getByText("WNBA")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /women.?s basketball/i }),
    ).toBeInTheDocument();
  });

  it("renders NBA hero copy", () => {
    render(<LeagueHero league="nba" />);
    expect(
      screen.getByRole("heading", { name: /men.?s basketball/i }),
    ).toBeInTheDocument();
  });
});
