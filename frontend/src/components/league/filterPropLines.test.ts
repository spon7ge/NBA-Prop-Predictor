import { describe, expect, it } from "vitest";
import type { ApiWnbaGame, ApiWnbaPropLine } from "@/lib/api";
import {
  collectStatOptions,
  collectTeamOptions,
  excludePropsFromFinalGames,
  excludePastGameProps,
  filterPropLines,
  nextEtDate,
  tipEtDate,
  type PropFilterSelection,
} from "./filterPropLines";

function prop(partial: Partial<ApiWnbaPropLine> & Pick<ApiWnbaPropLine, "player_name" | "stat" | "side">): ApiWnbaPropLine {
  return {
    team_abbrev: null,
    logo_url: null,
    market_type: "player_assists",
    game_date: null,
    commence_time: null,
    model_prediction: null,
    over_under_pct: null,
    ev: null,
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
  books: new Set(),
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
      ...empty,
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

  it("keeps rows that have a quote on any selected book", () => {
    const withBooks = [
      prop({
        player_name: "PP only",
        team_abbrev: "ATL",
        stat: "Points",
        side: "over",
        prizepicks: { line: 20, price: -110, is_dfs_flat_payout: true },
      }),
      prop({
        player_name: "FD only",
        team_abbrev: "SEA",
        stat: "Points",
        side: "over",
        fanduel: { line: 18, price: -115 },
      }),
      prop({
        player_name: "No books",
        team_abbrev: "NYL",
        stat: "Points",
        side: "over",
      }),
    ];
    const out = filterPropLines(withBooks, {
      ...empty,
      books: new Set(["prizepicks", "underdog"]),
    });
    expect(out.map((r) => r.player_name)).toEqual(["PP only"]);
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

describe("excludePastGameProps", () => {
  it("removes final-game teams and prior-day tips", () => {
    const games = [
      game({ status: "final", homeAbbrev: "ATL", awayAbbrev: "SEA" }),
      game({
        status: "scheduled",
        homeAbbrev: "WAS",
        awayAbbrev: "DAL",
        id: "later",
      }),
    ];
    const slateProps = [
      prop({
        player_name: "Yesterday",
        team_abbrev: "MIN",
        logo_url: "min.png",
        stat: "Points",
        side: "over",
        market_type: "player_points",
        commence_time: "2026-07-30T23:00:00Z",
      }),
      prop({
        player_name: "Final ATL",
        team_abbrev: "ATL",
        logo_url: "atl.png",
        stat: "Points",
        side: "over",
        market_type: "player_points",
        commence_time: "2026-07-31T23:30:00Z",
      }),
      prop({
        player_name: "Still on",
        team_abbrev: "WAS",
        logo_url: "was.png",
        stat: "Points",
        side: "over",
        market_type: "player_points",
        commence_time: "2026-07-31T23:30:00Z",
      }),
      prop({
        player_name: "Tomorrow",
        team_abbrev: "NYL",
        logo_url: "nyl.png",
        stat: "Points",
        side: "over",
        market_type: "player_points",
        commence_time: "2026-08-01T23:00:00Z",
      }),
    ];
    const out = excludePastGameProps(slateProps, games, "2026-07-31");
    expect(out.map((r) => r.player_name)).toEqual(["Still on", "Tomorrow"]);
  });

  it("keeps tomorrow tips when today's slate is entirely final", () => {
    const games = [
      game({ status: "final", homeAbbrev: "ATL", awayAbbrev: "SEA" }),
      game({
        status: "final",
        homeAbbrev: "WAS",
        awayAbbrev: "DAL",
        id: "g2",
      }),
    ];
    const slateProps = [
      prop({
        player_name: "Today Player",
        team_abbrev: "ATL",
        logo_url: "atl.png",
        stat: "Points",
        side: "over",
        commence_time: "2026-07-31T23:30:00Z",
      }),
      prop({
        player_name: "Tomorrow Player",
        team_abbrev: "NYL",
        logo_url: "nyl.png",
        stat: "Points",
        side: "over",
        market_type: "player_points",
        commence_time: "2026-08-01T23:00:00Z",
      }),
    ];
    const out = excludePastGameProps(slateProps, games, "2026-07-31");
    expect(out.map((r) => r.player_name)).toEqual(["Tomorrow Player"]);
  });
});

describe("tipEtDate / nextEtDate", () => {
  it("maps commence times into ET calendar dates", () => {
    expect(tipEtDate("2026-08-01T02:00:00Z")).toBe("2026-07-31");
    expect(tipEtDate("2026-08-01T23:00:00Z")).toBe("2026-08-01");
  });

  it("advances YYYY-MM-DD by one day", () => {
    expect(nextEtDate("2026-07-31")).toBe("2026-08-01");
  });
});
