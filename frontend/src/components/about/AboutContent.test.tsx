import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AboutContent } from "./AboutContent";

describe("AboutContent", () => {
  it("renders badge, headline, league pills, body copy, and later sections", () => {
    render(<AboutContent />);

    expect(screen.queryByRole("main")).not.toBeInTheDocument();
    expect(screen.getByText(/sports analytics/i)).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /about hoopvista/i }),
    ).toBeInTheDocument();
    const leaguePills = screen.getByRole("list", { name: "Leagues" });
    expect(leaguePills).toHaveTextContent("NBA");
    expect(leaguePills).toHaveTextContent("WNBA");
    expect(
      screen.getByText(/basketball analytics/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/plain-language/i)).toBeInTheDocument();
    expect(screen.getByText(/still in beta/i)).toBeInTheDocument();
    expect(screen.queryByText(/contributors/i)).not.toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Tech stack" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Data sources" }),
    ).toBeInTheDocument();
  });
});
