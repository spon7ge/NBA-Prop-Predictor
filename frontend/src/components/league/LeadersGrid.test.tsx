import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LeadersGrid } from "./LeadersGrid";
import type { ApiWnbaLeaderCategory } from "@/lib/api";

const categories: ApiWnbaLeaderCategory[] = [
  {
    key: "points",
    label: "Points",
    stat: "PTS",
    leaders: [
      {
        rank: 1,
        player_id: "1",
        name: "A'ja Wilson",
        team_abbrev: "LVA",
        gp: 25,
        value: "26.2",
      },
    ],
  },
  {
    key: "rebounds",
    label: "Rebounds",
    stat: "REB",
    leaders: [],
  },
];

describe("LeadersGrid", () => {
  it("renders season label, cards, colors, and attribution", () => {
    render(
      <LeadersGrid season={2026} categories={categories} />,
    );
    expect(screen.getByText("2026 season · per game")).toBeInTheDocument();
    expect(screen.getByText("Points")).toBeInTheDocument();
    expect(screen.getByText("A'ja Wilson")).toBeInTheDocument();
    expect(screen.getByText("LVA")).toBeInTheDocument();
    expect(screen.getByText("26.2")).toBeInTheDocument();
    expect(screen.getByText("No data")).toBeInTheDocument();
    expect(screen.getByText("Data: stats.wnba.com")).toBeInTheDocument();
  });

  it("shows loading skeletons", () => {
    render(<LeadersGrid season={2026} categories={[]} isLoading />);
    expect(screen.getByLabelText(/loading leaders/i)).toBeInTheDocument();
  });

  it("shows error copy when never loaded", () => {
    render(
      <LeadersGrid season={2026} categories={[]} isError />,
    );
    expect(screen.getByText(/leaders unavailable/i)).toBeInTheDocument();
  });
});
