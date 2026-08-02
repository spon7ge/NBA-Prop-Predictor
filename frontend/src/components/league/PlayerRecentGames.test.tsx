import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { ApiWnbaPlayerGame } from "@/lib/api";
import { PlayerRecentGames } from "./PlayerRecentGames";

function makeGame(index: number): ApiWnbaPlayerGame {
  return {
    game_id: String(index + 1),
    game_date: `2026-07-${String(index + 1).padStart(2, "0")}`,
    matchup: `LVA vs. T${index}`,
    min: "32",
    pts: String(20 + index),
    fg: "8-16",
    three_pt: "1-3",
    ft: "3-4",
    reb: "8",
    ast: "2",
    to: "1",
    stl: "1",
    blk: "1",
  };
}

function bodyRows() {
  const table = screen.getByRole("table");
  const rows = within(table).getAllByRole("row");
  // Drop header row
  return rows.slice(1);
}

describe("PlayerRecentGames", () => {
  it("shows 5 rows by default and expands with See more / Show less", async () => {
    const user = userEvent.setup();
    const games = Array.from({ length: 6 }, (_, i) => makeGame(i));

    render(<PlayerRecentGames games={games} />);

    expect(screen.getByText("Recent games")).toBeInTheDocument();
    expect(bodyRows()).toHaveLength(5);
    expect(screen.queryByText("LVA vs. T5")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "See more" }));

    expect(bodyRows()).toHaveLength(6);
    expect(screen.getByText("LVA vs. T5")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show less" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show less" }));
    expect(bodyRows()).toHaveLength(5);
  });

  it("hides See more when there are 5 or fewer games", () => {
    render(<PlayerRecentGames games={Array.from({ length: 3 }, (_, i) => makeGame(i))} />);

    expect(bodyRows()).toHaveLength(3);
    expect(screen.queryByRole("button", { name: "See more" })).not.toBeInTheDocument();
  });

  it("shows empty state when there are no games", () => {
    render(<PlayerRecentGames games={[]} />);

    expect(screen.getByText("No games yet")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "See more" })).not.toBeInTheDocument();
  });
});
