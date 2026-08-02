import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ApiWnbaPlayerResponse } from "@/lib/api";
import { PlayerHeader } from "./PlayerHeader";

const player: ApiWnbaPlayerResponse = {
  player_id: "1628932",
  name: "A'ja Wilson",
  position: "C",
  team_name: "Las Vegas Aces",
  team_abbrev: "LVA",
  headshot_url: "https://cdn.example.com/1628932.png",
  season: 2026,
  averages: {
    pts: "26.2",
    reb: "10.1",
    ast: "2.5",
    fg_pct: "52.0",
    fg3_pct: "33.0",
  },
  games: [],
  source_label: "stats.wnba.com",
};

describe("PlayerHeader", () => {
  it("renders bio and season averages", () => {
    render(<PlayerHeader player={player} />);

    expect(screen.getByText("A'ja Wilson")).toBeInTheDocument();
    expect(screen.getByText("C")).toBeInTheDocument();
    expect(screen.getByText("Las Vegas Aces")).toBeInTheDocument();

    expect(screen.getByText("PTS")).toBeInTheDocument();
    expect(screen.getByText("REB")).toBeInTheDocument();
    expect(screen.getByText("AST")).toBeInTheDocument();
    expect(screen.getByText("FG%")).toBeInTheDocument();
    expect(screen.getByText("3P%")).toBeInTheDocument();

    expect(screen.getByText("26.2")).toBeInTheDocument();
    expect(screen.getByText("10.1")).toBeInTheDocument();
    expect(screen.getByText("2.5")).toBeInTheDocument();
    expect(screen.getByText("52.0")).toBeInTheDocument();
    expect(screen.getByText("33.0")).toBeInTheDocument();
  });

  it("keeps a placeholder after headshot load error", () => {
    render(<PlayerHeader player={player} />);

    const img = screen.getByRole("img", { name: /A'ja Wilson/i });
    fireEvent.error(img);

    expect(
      screen.getByRole("img", { name: /placeholder/i }),
    ).toBeInTheDocument();
  });
});
