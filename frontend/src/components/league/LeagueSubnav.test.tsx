import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { LeagueSubnav } from "./LeagueSubnav";

describe("LeagueSubnav", () => {
  it("links Matchups and Leaders on WNBA; disables others", () => {
    render(
      <MemoryRouter initialEntries={["/wnba/leaders"]}>
        <LeagueSubnav league="wnba" />
      </MemoryRouter>,
    );
    const leaders = screen.getByRole("link", { name: "Leaders" });
    expect(leaders).toHaveAttribute("href", "/wnba/leaders");
    expect(leaders).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
      "href",
      "/wnba/matchups",
    );
    expect(
      screen.getByRole("button", { name: "HoopVista Picks" }),
    ).toBeDisabled();
  });

  it("keeps Leaders disabled on NBA", () => {
    render(
      <MemoryRouter initialEntries={["/nba/matchups"]}>
        <LeagueSubnav league="nba" />
      </MemoryRouter>,
    );
    expect(screen.getByRole("button", { name: "Leaders" })).toBeDisabled();
    expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
      "href",
      "/nba/matchups",
    );
  });
});
