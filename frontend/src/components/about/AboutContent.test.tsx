import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AboutContent } from "./AboutContent";

describe("AboutContent", () => {
  it("renders badge, headline, league pills, and body copy", () => {
    render(<AboutContent />);

    expect(screen.queryByRole("main")).not.toBeInTheDocument();
    expect(screen.getByText(/sports analytics/i)).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /about hoopvista/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("NBA")).toBeInTheDocument();
    expect(screen.getByText("WNBA")).toBeInTheDocument();
    expect(
      screen.getByText(/basketball analytics/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/plain-language/i)).toBeInTheDocument();
    expect(screen.getByText(/still in beta/i)).toBeInTheDocument();
    expect(screen.queryByText(/contributors/i)).not.toBeInTheDocument();
  });
});
