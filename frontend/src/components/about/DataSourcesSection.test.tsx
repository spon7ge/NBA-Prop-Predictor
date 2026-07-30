import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DataSourcesSection } from "./DataSourcesSection";

describe("DataSourcesSection", () => {
  it("renders sources with badges and external links", () => {
    render(<DataSourcesSection />);

    expect(
      screen.getByRole("heading", { name: "Data sources" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/public or licensed/i)).toBeInTheDocument();

    const nbaStats = screen.getByRole("link", { name: /NBA Stats API/i });
    expect(nbaStats).toHaveAttribute(
      "href",
      "https://github.com/swar/nba_api",
    );
    expect(nbaStats).toHaveAttribute("target", "_blank");
    expect(nbaStats).toHaveAttribute("rel", "noopener noreferrer");

    expect(
      screen.getByRole("link", { name: /The Odds API/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /Basketball-Reference/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /Supabase/i }),
    ).toBeInTheDocument();
  });
});
