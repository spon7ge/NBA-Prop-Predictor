import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import type { ApiWnbaLeaderCategory } from "@/lib/api";
import { LeaderCategoryCard } from "./LeaderCategoryCard";

const category: ApiWnbaLeaderCategory = {
  key: "points",
  label: "Points",
  stat: "PTS",
  leaders: [
    {
      rank: 1,
      player_id: "1628932",
      name: "A'ja Wilson",
      team_abbrev: "LVA",
      gp: 25,
      value: "26.2",
    },
  ],
};

describe("LeaderCategoryCard", () => {
  it("links the player name to the player profile", () => {
    render(
      <MemoryRouter>
        <LeaderCategoryCard category={category} />
      </MemoryRouter>,
    );

    expect(screen.getByRole("link", { name: "A'ja Wilson" })).toHaveAttribute(
      "href",
      "/wnba/player/1628932",
    );
  });
});
