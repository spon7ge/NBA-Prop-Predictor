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
      screen.getByRole("button", { name: "HoopVista Picks" }),
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
});
