import { describe, expect, it } from "vitest";
import type { ApiWnbaGame, ApiWnbaPropLine } from "@/lib/api";
import {
  collectStatOptions,
  collectTeamOptions,
  excludePropsFromFinalGames,
  filterPropLines,
  type PropFilterSelection,
} from "./filterPropLines";

function prop(partial: Partial<ApiWnbaPropLine> & Pick<ApiWnbaPropLine, "player_name" | "stat" | "side">): ApiWnbaPropLine {
  return {
    team_abbrev: null,
    logo_url: null,
    market_type: "player_assists",
    model_prediction: null,
    over_under_pct: null,
    ev: null,
    fanduel: null,
    draftkings: null,
    prizepicks: null,
    underdog: null,
    ...partial,
  };
}

const rows: ApiWnbaPropLine[] = [
  prop({
    player_name: "Rhyne Howard",
    team_abbrev: "ATL",
    logo_url: "atl.png",
    stat: "Assists",
    side: "over",
  }),
  prop({
    player_name: "Rhyne Howard",
    team_abbrev: "ATL",
    logo_url: "atl.png",
    stat: "Assists",
    side: "under",
  }),
  prop({
    player_name: "Jewell Loyd",
    team_abbrev: "SEA",
    logo_url: "sea.png",
    stat: "Points",
    side: "over",
    market_type: "player_points",
  }),
  prop({
    player_name: "Unknown",
    team_abbrev: null,
    logo_url: null,
    stat: "Points",
    side: "under",
    market_type: "player_points",
  }),
];

const empty: PropFilterSelection = {
  stats: new Set(),
  sides: new Set(),
  teams: new Set(),
};

describe("filterPropLines", () => {
  it("returns all rows when all filters are empty", () => {
    expect(filterPropLines(rows, empty)).toEqual(rows);
  });

  it("filters by stat with OR within the set", () => {
    const out = filterPropLines(rows, {
      ...empty,
      stats: new Set(["Assists"]),
    });
    expect(out.map((r) => r.stat)).toEqual(["Assists", "Assists"]);
  });

  it("filters by side", () => {
    const out = filterPropLines(rows, {
      ...empty,
      sides: new Set(["over"]),
    });
    expect(out.every((r) => r.side === "over")).toBe(true);
    expect(out).toHaveLength(2);
  });

  it("ANDs across filters", () => {
    const out = filterPropLines(rows, {
      stats: new Set(["Assists", "Points"]),
      sides: new Set(["over"]),
      teams: new Set(["SEA"]),
    });
    expect(out).toHaveLength(1);
    expect(out[0]?.player_name).toBe("Jewell Loyd");
  });

  it("excludes null team rows when Team filter is active", () => {
    const out = filterPropLines(rows, {
      ...empty,
      teams: new Set(["ATL", "SEA"]),
    });
    expect(out.every((r) => r.team_abbrev != null)).toBe(true);
    expect(out).toHaveLength(3);
  });
});

describe("collectStatOptions / collectTeamOptions", () => {
  it("collects sorted unique stats", () => {
    expect(collectStatOptions(rows)).toEqual(["Assists", "Points"]);
  });

  it("collects sorted unique teams with logos", () => {
    expect(collectTeamOptions(rows)).toEqual([
      { abbrev: "ATL", logoUrl: "atl.png" },
      { abbrev: "SEA", logoUrl: "sea.png" },
    ]);
  });
});

function game(
  partial: Partial<ApiWnbaGame> & {
    status: ApiWnbaGame["status"];
    homeAbbrev: string;
    awayAbbrev: string;
  },
): ApiWnbaGame {
  return {
    id: partial.id ?? "g1",
    espn_event_id: null,
    league: "wnba",
    status: partial.status,
    status_label: partial.status_label ?? partial.status,
    start_time_et: "7:00 PM ET",
    away: {
      abbrev: partial.awayAbbrev,
      name: partial.awayAbbrev,
      score: null,
      logo_url: null,
    },
    home: {
      abbrev: partial.homeAbbrev,
      name: partial.homeAbbrev,
      score: null,
      logo_url: null,
    },
  };
}

describe("excludePropsFromFinalGames", () => {
  it("removes props for both teams in a final game", () => {
    const games = [game({ status: "final", homeAbbrev: "ATL", awayAbbrev: "SEA" })];
    const out = excludePropsFromFinalGames(rows, games);
    expect(out.map((r) => r.player_name)).toEqual(["Unknown"]);
  });

  it("keeps props for live and scheduled games", () => {
    const games = [
      game({ status: "live", homeAbbrev: "ATL", awayAbbrev: "CHI", id: "live" }),
      game({
        status: "scheduled",
        homeAbbrev: "SEA",
        awayAbbrev: "LAS",
        id: "sched",
      }),
    ];
    const out = excludePropsFromFinalGames(rows, games);
    expect(out).toEqual(rows);
  });

  it("keeps rows with null team_abbrev", () => {
    const games = [game({ status: "final", homeAbbrev: "ATL", awayAbbrev: "SEA" })];
    const out = excludePropsFromFinalGames(rows, games);
    expect(out.some((r) => r.team_abbrev == null)).toBe(true);
  });

  it("does not filter when games are empty or undefined", () => {
    expect(excludePropsFromFinalGames(rows, [])).toEqual(rows);
    expect(excludePropsFromFinalGames(rows, undefined)).toEqual(rows);
    expect(excludePropsFromFinalGames(rows, null)).toEqual(rows);
  });
});
