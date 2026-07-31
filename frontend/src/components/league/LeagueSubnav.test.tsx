import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { LeagueSubnav } from "./LeagueSubnav";

describe("LeagueSubnav", () => {
  it("links Matchups, Leaders, and Standings on WNBA; disables others", () => {
    render(
      <MemoryRouter initialEntries={["/wnba/standings"]}>
        <LeagueSubnav league="wnba" />
      </MemoryRouter>,
    );
    const standings = screen.getByRole("link", { name: "Standings" });
    expect(standings).toHaveAttribute("href", "/wnba/standings");
    expect(standings).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Leaders" })).toHaveAttribute(
      "href",
      "/wnba/leaders",
    );
    expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
      "href",
      "/wnba/matchups",
    );
    expect(
      screen.getByRole("button", { name: "Prop Picks" }),
    ).toBeDisabled();
  });

  it("keeps Leaders and Standings disabled on NBA", () => {
    render(
      <MemoryRouter initialEntries={["/nba/matchups"]}>
        <LeagueSubnav league="nba" />
      </MemoryRouter>,
    );
    expect(screen.getByRole("button", { name: "Leaders" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Standings" })).toBeDisabled();
    expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
      "href",
      "/nba/matchups",
    );
  });

  it("places Explore and Learn labels inline with a divider before Learn", () => {
    render(
      <MemoryRouter initialEntries={["/wnba/matchups"]}>
        <LeagueSubnav league="wnba" />
      </MemoryRouter>,
    );
    expect(screen.getByText("Explore")).toBeInTheDocument();
    expect(screen.getByText("Learn")).toBeInTheDocument();
    const learnGroup = screen.getByText("Learn").closest("div");
    expect(learnGroup?.className).toMatch(/border-l/);
    // Labels are siblings of the pill row inside the same flex group
    expect(learnGroup?.className).toMatch(/items-center/);
    const exploreGroup = screen.getByText("Explore").closest("div");
    expect(exploreGroup?.className).toMatch(/items-center/);
    // Smoke: nav still works
    expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
      "href",
      "/wnba/matchups",
    );
  });
});
