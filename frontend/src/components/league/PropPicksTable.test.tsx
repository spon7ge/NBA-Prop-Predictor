import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { ApiWnbaPropLine } from "@/lib/api";
import { PROP_PICKS_PAGE_SIZE, PropPicksTable } from "./PropPicksTable";

const emptyBooks = {
  fanduel: null,
  draftkings: null,
  caesars: null,
  betmgm: null,
  pinnacle: null,
  bet365: null,
  prizepicks: null,
  underdog: null,
  betr: null,
  novig: null,
  sleeper: null,
  pick6: null,
} as const;

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
    game_date: "2026-07-31",
    commence_time: "2026-07-31T23:30:00Z",
    ...emptyBooks,
    fanduel: { line: 3.5, odds_american: -114 },
    draftkings: { line: 3.5, odds_american: -120 },
    betmgm: { line: 3.5, odds_american: -115 },
    pinnacle: { line: 3.5, odds_american: -108 },
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
    game_date: "2026-07-31",
    commence_time: "2026-07-31T23:30:00Z",
    ...emptyBooks,
    fanduel: { line: 3.5, odds_american: -114 },
    draftkings: { line: 3.5, odds_american: -110 },
    pinnacle: { line: 3.5, odds_american: -112 },
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
    expect(screen.getByRole("columnheader", { name: "Caesars" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "BetMGM" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Pinnacle" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "bet365" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "PrizePicks" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Underdog" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Betr" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Novig" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Sleeper" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Pick6" })).toBeInTheDocument();

    expect(screen.getAllByText("Rhyne Howard")).toHaveLength(2);
    expect(screen.getAllByRole("presentation")).toHaveLength(2);
    expect(screen.getByText("Over")).toBeInTheDocument();
    expect(screen.getByText("Under")).toBeInTheDocument();
    expect(screen.getByText("Odds by Parlay API")).toBeInTheDocument();
  });

  it("shows line only when odds_american is null", () => {
    render(
      <PropPicksTable
        props={[
          {
            ...sampleProps[0]!,
            ...emptyBooks,
            fanduel: { line: 4.5, odds_american: null },
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

  it("shows custom empty message when provided", () => {
    render(
      <PropPicksTable
        props={[]}
        emptyMessage="No active props — today's games are final"
      />,
    );
    expect(
      screen.getByText("No active props — today's games are final"),
    ).toBeInTheDocument();
  });

  it("shows loading skeletons", () => {
    render(<PropPicksTable props={[]} isLoading />);
    expect(screen.getByLabelText("Loading prop picks")).toBeInTheDocument();
  });

  it("shows last updated when lastUpdatedAt is set", () => {
    render(
      <PropPicksTable
        props={sampleProps}
        lastUpdatedAt={Date.parse("2026-07-31T23:54:00")}
      />,
    );

    expect(screen.getByText(/Last updated/i)).toBeInTheDocument();
    expect(screen.getByText(/Jul 31/i)).toBeInTheDocument();
  });

  it("hides last updated when lastUpdatedAt is unset", () => {
    render(<PropPicksTable props={sampleProps} />);
    expect(screen.queryByText(/Last updated/i)).not.toBeInTheDocument();
  });

  it("paginates to 50 rows with next/previous", async () => {
    const user = userEvent.setup();
    const many: ApiWnbaPropLine[] = Array.from(
      { length: PROP_PICKS_PAGE_SIZE + 3 },
      (_, i) => ({
        ...sampleProps[0]!,
        player_name: `Player ${i}`,
        market_type: `player_points_${i}`,
      }),
    );

    render(<PropPicksTable props={many} />);

    expect(screen.getByText("Showing 1–50 of 53")).toBeInTheDocument();
    expect(screen.getByText("Player 0")).toBeInTheDocument();
    expect(screen.queryByText("Player 50")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("Showing 51–53 of 53")).toBeInTheDocument();
    expect(screen.getByText("Player 50")).toBeInTheDocument();
    expect(screen.queryByText("Player 0")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  });
});