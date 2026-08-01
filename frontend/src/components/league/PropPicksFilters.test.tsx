import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { PropPicksFilters } from "./PropPicksFilters";

describe("PropPicksFilters", () => {
  it("renders book/stat/team filters, disabled +EV Soon, and clear when active", async () => {
    const user = userEvent.setup();
    const onClear = vi.fn();
    const onStatsChange = vi.fn();
    const onBooksChange = vi.fn();

    const { rerender } = render(
      <PropPicksFilters
        stats={["Assists", "Points"]}
        teams={[
          { abbrev: "ATL", logoUrl: "atl.png" },
          { abbrev: "SEA", logoUrl: null },
        ]}
        selectedStats={new Set()}
        selectedSides={new Set()}
        selectedTeams={new Set()}
        selectedBooks={new Set()}
        onStatsChange={onStatsChange}
        onSidesChange={vi.fn()}
        onTeamsChange={vi.fn()}
        onBooksChange={onBooksChange}
        onClear={onClear}
      />,
    );

    expect(screen.getByRole("button", { name: "Book" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stat" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "O/U" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Team" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "+EV · Soon" }),
    ).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Clear filters" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Book" }));
    expect(
      screen.getByRole("option", { name: /PrizePicks/i }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: /PrizePicks/i }));
    expect(onBooksChange).toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Stat" }));
    expect(screen.getByRole("option", { name: /Assists/i })).toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: /Assists/i }));
    expect(onStatsChange).toHaveBeenCalled();

    rerender(
      <PropPicksFilters
        stats={["Assists", "Points"]}
        teams={[
          { abbrev: "ATL", logoUrl: "atl.png" },
          { abbrev: "SEA", logoUrl: null },
        ]}
        selectedStats={new Set(["Assists"])}
        selectedSides={new Set()}
        selectedTeams={new Set()}
        selectedBooks={new Set(["prizepicks"])}
        onStatsChange={onStatsChange}
        onSidesChange={vi.fn()}
        onTeamsChange={vi.fn()}
        onBooksChange={onBooksChange}
        onClear={onClear}
      />,
    );

    expect(screen.getByRole("button", { name: "Stat (1)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Book (1)" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(onClear).toHaveBeenCalled();
  });
});
