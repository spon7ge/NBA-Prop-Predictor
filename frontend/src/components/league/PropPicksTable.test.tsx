import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ApiWnbaPropLine } from "@/lib/api";
import { PropPicksTable } from "./PropPicksTable";

const sampleProps: ApiWnbaPropLine[] = [
  {
    player_name: "Rhyne Howard",
    stat: "Assists",
    market_type: "player_assists",
    side: "over",
    model_prediction: null,
    over_under_pct: null,
    ev: null,
    fanduel: { line: 3.5, odds_american: -114 },
    draftkings: { line: 3.5, odds_american: -120 },
  },
  {
    player_name: "Rhyne Howard",
    stat: "Assists",
    market_type: "player_assists",
    side: "under",
    model_prediction: null,
    over_under_pct: null,
    ev: null,
    fanduel: { line: 3.5, odds_american: -114 },
    draftkings: { line: 3.5, odds_american: -110 },
  },
];

describe("PropPicksTable", () => {
  it("renders player, stat, both sides, and book pills", () => {
    render(<PropPicksTable props={sampleProps} />);

    expect(screen.getByRole("columnheader", { name: "Player" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Model" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "O/U%" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "EV" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "FanDuel" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "DraftKings" })).toBeInTheDocument();

    expect(screen.getAllByText("Rhyne Howard")).toHaveLength(2);
    expect(screen.getByText("Over")).toBeInTheDocument();
    expect(screen.getByText("Under")).toBeInTheDocument();
    expect(screen.getAllByText("3.5 −114").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("3.5 −120")).toBeInTheDocument();
    expect(screen.getByText("3.5 −110")).toBeInTheDocument();
    expect(screen.getByText("Odds by FanDuel & DraftKings")).toBeInTheDocument();
  });

  it("shows unavailable copy when empty or error", () => {
    const { rerender } = render(<PropPicksTable props={[]} />);
    expect(screen.getByText("Prop lines unavailable")).toBeInTheDocument();

    rerender(<PropPicksTable props={sampleProps} isError />);
    expect(screen.getByText("Prop lines unavailable")).toBeInTheDocument();
  });

  it("shows loading skeletons", () => {
    render(<PropPicksTable props={[]} isLoading />);
    expect(screen.getByLabelText("Loading prop picks")).toBeInTheDocument();
  });
});
