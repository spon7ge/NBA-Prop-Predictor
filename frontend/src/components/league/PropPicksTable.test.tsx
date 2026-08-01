import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ApiWnbaPropLine } from "@/lib/api";
import { PropPicksTable } from "./PropPicksTable";

const sampleProps: ApiWnbaPropLine[] = [
  {
    player_name: "Rhyne Howard",
    team_abbrev: "ATL",
    logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    stat: "Assists",
    market_type: "player_assists",
    side: "over",
    model_prediction: null,
    over_under_pct: null,
    ev: null,
    fanduel: { line: 3.5, odds_american: -114 },
    draftkings: { line: 3.5, odds_american: -120 },
    prizepicks: { line: 3.5, odds_american: null },
    underdog: { line: 3.5, odds_american: -108 },
  },
  {
    player_name: "Rhyne Howard",
    team_abbrev: "ATL",
    logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    stat: "Assists",
    market_type: "player_assists",
    side: "under",
    model_prediction: null,
    over_under_pct: null,
    ev: null,
    fanduel: { line: 3.5, odds_american: -114 },
    draftkings: { line: 3.5, odds_american: -110 },
    prizepicks: null,
    underdog: null,
  },
];

describe("PropPicksTable", () => {
  it("renders player, team logo, stat, both sides, and book pills", () => {
    render(<PropPicksTable props={sampleProps} />);

    expect(screen.getByRole("columnheader", { name: "Player" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Team" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Model" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "O/U%" })).not.toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "EV" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "FanDuel" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "DraftKings" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "PrizePicks" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Underdog" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "BetMGM" })).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "BetRivers" })).not.toBeInTheDocument();

    expect(screen.getAllByText("Rhyne Howard")).toHaveLength(2);
    expect(screen.getAllByRole("presentation")).toHaveLength(2);
    expect(screen.getAllByRole("presentation")[0]).toHaveAttribute(
      "src",
      "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    );
    expect(screen.getByText("Over")).toBeInTheDocument();
    expect(screen.getByText("Under")).toBeInTheDocument();
    // Line above odds in each book pill (PrizePicks over row has line only)
    expect(screen.getAllByText("3.5").length).toBeGreaterThanOrEqual(5);
    expect(screen.getAllByText("−114").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("−120")).toBeInTheDocument();
    expect(screen.getByText("−110")).toBeInTheDocument();
    expect(screen.getByText("−108")).toBeInTheDocument();
    expect(
      screen.getByText("Odds by FanDuel, DraftKings, PrizePicks & Underdog"),
    ).toBeInTheDocument();
  });

  it("shows line only when odds_american is null", () => {
    render(
      <PropPicksTable
        props={[
          {
            ...sampleProps[0]!,
            fanduel: { line: 4.5, odds_american: null },
            draftkings: null,
            prizepicks: null,
            underdog: null,
          },
        ]}
      />,
    );

    expect(screen.getByText("4.5")).toBeInTheDocument();
    expect(screen.queryByText("−114")).not.toBeInTheDocument();
  });

  it("shows unavailable copy when empty or error", () => {
    const { rerender } = render(<PropPicksTable props={[]} />);
    expect(screen.getByText("Prop lines unavailable")).toBeInTheDocument();

    rerender(<PropPicksTable props={sampleProps} isError />);
    expect(screen.getByText("Prop lines unavailable")).toBeInTheDocument();
  });

  it("shows filter-empty copy when filters hide all rows", () => {
    render(<PropPicksTable props={[]} filtersActive />);
    expect(screen.getByText("No props match these filters")).toBeInTheDocument();
  });

  it("shows loading skeletons", () => {
    render(<PropPicksTable props={[]} isLoading />);
    expect(screen.getByLabelText("Loading prop picks")).toBeInTheDocument();
  });
});
