import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  DEFAULT_LEARN_SPORTS,
  LearnTheGameSection,
} from "./LearnTheGameSection";
import type { LearnSport } from "./types";

describe("LearnTheGameSection", () => {
  it("renders default NBA and WNBA primers", () => {
    render(<LearnTheGameSection />);
    expect(
      screen.getByRole("heading", { name: "Learn the Game" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("New to a sport? Start here."),
    ).toBeInTheDocument();
    expect(screen.getAllByText("How it works →")).toHaveLength(
      DEFAULT_LEARN_SPORTS.length,
    );
    expect(screen.getByText("NBA")).toBeInTheDocument();
    expect(screen.getByText("WNBA")).toBeInTheDocument();
  });

  it("renders provided sports instead of defaults", () => {
    const sports: LearnSport[] = [
      {
        id: "custom",
        league: "nba",
        sport: "Custom Ball",
        href: "#custom",
      },
    ];
    render(<LearnTheGameSection sports={sports} />);
    expect(screen.getByText("Custom Ball")).toBeInTheDocument();
    expect(screen.getAllByText("How it works →")).toHaveLength(1);
  });

  it("shows empty copy when an empty list is passed", () => {
    render(<LearnTheGameSection sports={[]} />);
    expect(screen.getByText("No primers yet.")).toBeInTheDocument();
  });
});
