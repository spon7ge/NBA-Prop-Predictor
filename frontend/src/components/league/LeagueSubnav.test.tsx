import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LeagueSubnav } from "./LeagueSubnav";

describe("LeagueSubnav", () => {
  it("marks Matchups active and disables other items", () => {
    render(<LeagueSubnav league="wnba" />);
    expect(screen.getByRole("button", { name: "Matchups" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(
      screen.getByRole("button", { name: "HoopVista Picks" }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: "Leaders" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Glossary" })).toBeDisabled();
  });
});
