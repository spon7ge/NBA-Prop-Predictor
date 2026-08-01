import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { LeagueCtaSection } from "./LeagueCtaSection";

describe("LeagueCtaSection", () => {
  it("links into NBA and WNBA matchups", () => {
    render(
      <MemoryRouter>
        <LeagueCtaSection />
      </MemoryRouter>,
    );
    expect(
      screen.getByRole("heading", { name: /enter a league/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "NBA" })).toHaveAttribute(
      "href",
      "/nba/matchups",
    );
    expect(screen.getByRole("link", { name: "WNBA" })).toHaveAttribute(
      "href",
      "/wnba/matchups",
    );
  });
});
