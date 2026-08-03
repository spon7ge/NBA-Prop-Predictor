import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { MatchupGameCard } from "./MatchupGameCard";
import type { MatchupGame } from "./types";

const liveGame: MatchupGame = {
  id: "1",
  espnEventId: "401857098",
  league: "wnba",
  status: "live",
  statusLabel: "3:31 - 4th",
  venue: "Mortgage Matchup Center",
  venueCity: "Phoenix",
  away: {
    abbrev: "GS",
    name: "Golden State Valkyries",
    score: 77,
    record: "19-8",
    logoUrl: null,
  },
  home: {
    abbrev: "PHX",
    name: "Phoenix Mercury",
    score: 78,
    record: "10-18",
    logoUrl: null,
  },
};

function renderCard(game: MatchupGame) {
  return render(
    <MemoryRouter>
      <MatchupGameCard game={game} />
    </MemoryRouter>,
  );
}

describe("MatchupGameCard", () => {
  it("links to game detail when espnEventId is set", () => {
    renderCard(liveGame);
    expect(
      screen.getByRole("link", { name: /golden state valkyries/i }),
    ).toHaveAttribute("href", "/games/401857098");
  });

  it("links to MLB game stub when mlbGamePk is set", () => {
    renderCard({
      ...liveGame,
      id: "mlb-824971",
      espnEventId: null,
      mlbGamePk: "824971",
      league: "mlb",
      away: {
        abbrev: "NYY",
        name: "New York Yankees",
        score: 3,
        record: "55-35",
        logoUrl: null,
      },
      home: {
        abbrev: "BOS",
        name: "Boston Red Sox",
        score: 2,
        record: "50-40",
        logoUrl: null,
      },
    });
    expect(
      screen.getByRole("link", { name: /new york yankees/i }),
    ).toHaveAttribute("href", "/mlb/games/824971");
  });

  it("shows venue · city, status, records, and scores", () => {
    renderCard(liveGame);
    expect(screen.getByText("3:31 - 4th")).toBeInTheDocument();
    expect(
      screen.getByText("Mortgage Matchup Center · Phoenix"),
    ).toBeInTheDocument();
    expect(screen.getByText("19-8")).toBeInTheDocument();
    expect(screen.getByText("10-18")).toBeInTheDocument();
    expect(screen.getByText("77")).toBeInTheDocument();
    expect(screen.getByText("78")).toBeInTheDocument();
  });

  it("omits venue line and records when absent", () => {
    renderCard({
      ...liveGame,
      espnEventId: null,
      venue: null,
      venueCity: null,
      away: { ...liveGame.away, record: null },
      home: { ...liveGame.home, record: null },
    });
    expect(screen.queryByText(/Mortgage/)).not.toBeInTheDocument();
    expect(screen.queryByText("19-8")).not.toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("hides score badges for scheduled games only", () => {
    renderCard({
      ...liveGame,
      status: "scheduled",
      statusLabel: "8:00 PM ET",
      away: { ...liveGame.away, score: null },
      home: { ...liveGame.home, score: null },
    });

    expect(screen.queryByText("77")).not.toBeInTheDocument();
    expect(screen.queryByText("78")).not.toBeInTheDocument();
    expect(screen.queryByText("Golden State Valkyries")).toBeInTheDocument();
    expect(screen.queryByText("Phoenix Mercury")).toBeInTheDocument();
  });

  it("renders team logos when logoUrl is set", () => {
    const { container } = renderCard({
      ...liveGame,
      away: {
        ...liveGame.away,
        logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500/gs.png",
      },
      home: {
        ...liveGame.home,
        logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500/phx.png",
      },
    });
    const images = container.querySelectorAll("img");
    expect(images).toHaveLength(2);
    expect(images[0]).toHaveAttribute(
      "src",
      "https://a.espncdn.com/i/teamlogos/wnba/500/gs.png",
    );
  });

  it("renders the abbrev letter when logoUrl is null", () => {
    renderCard(liveGame);
    expect(screen.getByText("G")).toBeInTheDocument();
    expect(screen.queryByRole("presentation")).not.toBeInTheDocument();
  });

  it("shows DraftKings odds pill and caption when odds are present", () => {
    renderCard({
      ...liveGame,
      odds: {
        spreadTeamAbbrev: "ATL",
        spreadLine: -12.5,
        total: 178.5,
        sportsbook: "draftkings",
      },
    });
    expect(
      screen.getByText("Spread: ATL -12.5 · Total: 178.5"),
    ).toBeInTheDocument();
    expect(screen.getByText("Odds by")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "DraftKings" })).toBeInTheDocument();
    expect(screen.getByTestId("matchup-odds")).toHaveAttribute(
      "data-sportsbook",
      "draftkings",
    );
  });

  it("shows FanDuel caption when sportsbook is fanduel", () => {
    renderCard({
      ...liveGame,
      odds: {
        spreadTeamAbbrev: "DAL",
        spreadLine: -11.5,
        total: 171.5,
        sportsbook: "fanduel",
      },
    });
    expect(screen.getByText("Odds by")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "FanDuel" })).toBeInTheDocument();
    expect(
      screen.queryByRole("img", { name: "DraftKings" }),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("matchup-odds")).toHaveAttribute(
      "data-sportsbook",
      "fanduel",
    );
  });

  it("places odds under scores for live and final games", () => {
    const withOdds = {
      ...liveGame,
      odds: {
        spreadTeamAbbrev: "ATL" as const,
        spreadLine: -12.5,
        total: 178.5,
      },
    };

    const { unmount } = renderCard(withOdds);
    expect(screen.getByTestId("matchup-odds")).toHaveAttribute(
      "data-placement",
      "under-scores",
    );
    unmount();

    renderCard({
      ...withOdds,
      status: "final",
      statusLabel: "Final",
    });
    expect(screen.getByTestId("matchup-odds")).toHaveAttribute(
      "data-placement",
      "under-scores",
    );
  });

  it("keeps odds beside the home row for scheduled games", () => {
    renderCard({
      ...liveGame,
      status: "scheduled",
      statusLabel: "8:00 PM ET",
      away: { ...liveGame.away, score: null },
      home: { ...liveGame.home, score: null },
      odds: {
        spreadTeamAbbrev: "ATL",
        spreadLine: -12.5,
        total: 178.5,
      },
    });
    expect(screen.getByTestId("matchup-odds")).toHaveAttribute(
      "data-placement",
      "beside-home",
    );
  });

  it("shows a partial odds pill when only total is present", () => {
    renderCard({
      ...liveGame,
      odds: {
        spreadTeamAbbrev: null,
        spreadLine: null,
        total: 178.5,
      },
    });
    expect(screen.getByText("Total: 178.5")).toBeInTheDocument();
    expect(screen.getByText("Odds by")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "DraftKings" })).toBeInTheDocument();
  });

  it("omits odds pill when odds are absent", () => {
    renderCard({ ...liveGame, odds: null });
    expect(screen.queryByText(/Spread:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Total:/)).not.toBeInTheDocument();
    expect(screen.queryByText("Odds by")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("img", { name: "DraftKings" }),
    ).not.toBeInTheDocument();
  });
});
