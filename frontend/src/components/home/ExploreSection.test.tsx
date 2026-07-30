import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ExploreSection } from "./ExploreSection";
import type { ExploreItem } from "./types";

describe("ExploreSection", () => {
  it("renders default explore cards when no prop is provided", () => {
    render(<ExploreSection />);
    expect(
      screen.getByRole("heading", { name: "Explore" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Player props, explained" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Featured")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "WNBA clutch minutes" }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("article").length).toBeGreaterThanOrEqual(5);
  });

  it("renders skeleton placeholders when an empty list is passed", () => {
    const { container } = render(<ExploreSection items={[]} />);
    expect(
      screen.getByText("Browse leagues and topics — coming soon."),
    ).toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(5);
  });

  it("renders provided explore items", () => {
    const items: ExploreItem[] = [
      {
        id: "featured-nba",
        league: "nba",
        headline: "Custom featured topic",
        summary: "A custom summary for the featured card.",
        graphic: "chart",
        featured: true,
      },
      {
        id: "wnba-topic",
        league: "wnba",
        headline: "Custom WNBA topic",
        summary: "Another custom summary.",
        graphic: "bars",
      },
    ];
    render(<ExploreSection items={items} />);
    expect(
      screen.getByRole("heading", { name: "Custom featured topic" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Custom WNBA topic" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Player props, explained")).not.toBeInTheDocument();
  });
});
