import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DEFAULT_STORIES, StoriesSection } from "./StoriesSection";
import type { Story } from "./types";

describe("StoriesSection", () => {
  it("renders default stories when no prop is provided", () => {
    render(<StoriesSection />);
    expect(
      screen.getByRole("heading", { name: "Stories" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Summer League is over. Who won?"),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("article")).toHaveLength(DEFAULT_STORIES.length);
  });

  it("renders provided stories instead of defaults", () => {
    const stories: Story[] = [
      {
        id: "custom",
        league: "nba",
        headline: "Custom headline only",
        dateLabel: "JUL 29, 2026",
        summary: "A one-off story for tests.",
        graphic: "crown",
      },
    ];
    render(<StoriesSection stories={stories} />);
    expect(screen.getByText("Custom headline only")).toBeInTheDocument();
    expect(
      screen.queryByText("Summer League is over. Who won?"),
    ).not.toBeInTheDocument();
    expect(screen.getAllByRole("article")).toHaveLength(1);
  });

  it("shows empty copy when an empty list is passed", () => {
    render(<StoriesSection stories={[]} />);
    expect(screen.getByText("No stories yet.")).toBeInTheDocument();
    expect(screen.queryAllByRole("article")).toHaveLength(0);
  });
});
