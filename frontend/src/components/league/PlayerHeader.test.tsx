import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ApiWnbaPlayerResponse } from "@/lib/api";
import { PlayerHeader } from "./PlayerHeader";

const player: ApiWnbaPlayerResponse = {
  player_id: "1628932",
  name: "A'ja Wilson",
  position: "Center",
  jersey: "22",
  height: "6' 4\"",
  birthdate: "8/8/1996 (29)",
  college: "South Carolina",
  draft_info: "2018: Rd 1, Pk 1 (LVA)",
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
    expect(
      screen.getByText((content) => content.includes("#22") && content.includes("Center")),
    ).toBeInTheDocument();
    expect(
      screen.getByText((content) => content.includes("Las Vegas Aces")),
    ).toBeInTheDocument();

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

  it("renders ESPN-style bio facts", () => {
    render(<PlayerHeader player={player} />);
    expect(screen.getByText(/#22/)).toBeInTheDocument();
    expect(screen.getByText(/Center/)).toBeInTheDocument();
    expect(screen.getByText("Height")).toBeInTheDocument();
    expect(screen.getByText("6' 4\"")).toBeInTheDocument();
    expect(screen.getByText("Birthdate")).toBeInTheDocument();
    expect(screen.getByText("College")).toBeInTheDocument();
    expect(screen.getByText("South Carolina")).toBeInTheDocument();
    expect(screen.getByText("Draft Info")).toBeInTheDocument();
    expect(screen.getByText("2018: Rd 1, Pk 1 (LVA)")).toBeInTheDocument();
  });

  it("omits missing bio rows", () => {
    render(
      <PlayerHeader
        player={{
          ...player,
          height: null,
          college: null,
          draft_info: null,
          birthdate: null,
        }}
      />,
    );
    expect(screen.queryByText("Height")).not.toBeInTheDocument();
    expect(screen.queryByText("College")).not.toBeInTheDocument();
    expect(screen.queryByText("Draft Info")).not.toBeInTheDocument();
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
