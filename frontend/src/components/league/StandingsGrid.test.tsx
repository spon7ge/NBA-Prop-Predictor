import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StandingsGrid } from "./StandingsGrid";
import type { ApiWnbaStandingsConference } from "@/lib/api";

const sample: ApiWnbaStandingsConference[] = [
  {
    key: "east",
    label: "Eastern Conference",
    teams: [
      {
        rank: 1,
        team_id: "5",
        abbrev: "IND",
        name: "Indiana Fever",
        logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/ind.png",
        wins: 18,
        losses: 10,
        wl: "18-10",
        pct: ".643",
        gb: "-",
        home: "11-5",
        away: "7-5",
        l10: "8-2",
        diff: "+169",
        streak: "W4",
      },
    ],
  },
  {
    key: "west",
    label: "Western Conference",
    teams: [
      {
        rank: 1,
        team_id: "16",
        abbrev: "MIN",
        name: "Minnesota Lynx",
        logo_url: null,
        wins: 24,
        losses: 6,
        wl: "24-6",
        pct: ".800",
        gb: "-",
        home: "13-2",
        away: "11-4",
        l10: "8-2",
        diff: "-12",
        streak: "L2",
      },
    ],
  },
];

describe("StandingsGrid", () => {
  it("renders season, conferences, rows, and attribution", () => {
    render(
      <StandingsGrid season={2026} conferences={sample} />,
    );
    expect(screen.getByText("2026 regular season")).toBeInTheDocument();
    expect(screen.getByText("Eastern Conference")).toBeInTheDocument();
    expect(screen.getByText("Western Conference")).toBeInTheDocument();
    expect(screen.getByText("Indiana Fever")).toBeInTheDocument();
    expect(screen.getByText("IND")).toBeInTheDocument();
    expect(screen.getByText("18-10")).toBeInTheDocument();
    expect(screen.getByText("Data: ESPN")).toBeInTheDocument();
  });

  it("shows loading skeletons", () => {
    render(
      <StandingsGrid season={2026} conferences={[]} isLoading />,
    );
    expect(screen.getByLabelText("Loading standings")).toBeInTheDocument();
  });

  it("shows error copy when never loaded", () => {
    render(
      <StandingsGrid season={2026} conferences={[]} isError />,
    );
    expect(screen.getByText("Standings unavailable")).toBeInTheDocument();
  });

  it("shows No data for empty conference", () => {
    render(
      <StandingsGrid
        season={2026}
        conferences={[
          { key: "east", label: "Eastern Conference", teams: [] },
        ]}
      />,
    );
    expect(screen.getByText("No data")).toBeInTheDocument();
  });
});
