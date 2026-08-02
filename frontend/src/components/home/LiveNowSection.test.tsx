import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { LiveNowSection } from "./LiveNowSection";
import type { LiveGame } from "./types";

const liveGame: LiveGame = {
  id: "g1",
  league: "wnba",
  status: "live",
  statusLabel: "Q3 7:13",
  away: { abbrev: "ATL", name: "Atlanta Dream", score: 36, logoUrl: null },
  home: { abbrev: "DAL", name: "Dallas Wings", score: 44, logoUrl: null },
};

const linkedLiveGame: LiveGame = {
  ...liveGame,
  espnEventId: "401857098",
};

const finalGame: LiveGame = {
  id: "g2",
  league: "wnba",
  status: "final",
  statusLabel: "Final",
  away: { abbrev: "NYL", name: "New York Liberty", score: 90, logoUrl: null },
  home: { abbrev: "LAS", name: "Los Angeles Sparks", score: 80, logoUrl: null },
};

describe("LiveNowSection", () => {
  it("shows skeletons while loading with no games", () => {
    const { container } = render(<LiveNowSection isLoading games={[]} />);
    expect(screen.getByText("0 games in progress")).toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(3);
  });

  it("shows empty state without skeletons when loaded with zero games", () => {
    const { container } = render(<LiveNowSection isLoading={false} games={[]} />);
    expect(screen.getByText("0 games in progress")).toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(0);
  });

  it("shows only in-progress games and counts them in the subtitle", () => {
    render(<LiveNowSection games={[liveGame, finalGame]} />);
    expect(screen.getByText("1 game in progress")).toBeInTheDocument();
    expect(screen.getByText("ATL")).toBeInTheDocument();
    expect(screen.queryByText("NYL")).not.toBeInTheDocument();
  });

  it("shows a muted error when the scoreboard never loaded", () => {
    const { container } = render(<LiveNowSection isError games={[]} />);
    expect(screen.getByText("Unable to load scoreboard")).toBeInTheDocument();
    expect(container.querySelectorAll("article")).toHaveLength(0);
  });

  it("keeps showing games when an error follows a successful load", () => {
    render(<LiveNowSection isError={false} games={[liveGame]} />);
    expect(screen.queryByText("Unable to load scoreboard")).not.toBeInTheDocument();
    expect(screen.getByText("ATL")).toBeInTheDocument();
  });

  it("prefers skeletons over the error message while still loading", () => {
    const { container } = render(<LiveNowSection isError isLoading games={[]} />);
    expect(screen.queryByText("Unable to load scoreboard")).not.toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(3);
  });

  it("links to game detail when espnEventId is present", () => {
    render(
      <MemoryRouter>
        <LiveNowSection games={[linkedLiveGame]} />
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: /Atlanta Dream/i })).toHaveAttribute(
      "href",
      "/games/401857098",
    );
  });

  it("links MLB games to /mlb/games/:gamePk", () => {
    const mlbLiveGame: LiveGame = {
      id: "mlb-9",
      league: "mlb",
      espnEventId: null,
      mlbGamePk: "9",
      status: "live",
      statusLabel: "Top 3rd",
      away: {
        abbrev: "BOS",
        name: "Boston Red Sox",
        score: 2,
        logoUrl: null,
      },
      home: {
        abbrev: "NYY",
        name: "New York Yankees",
        score: 3,
        logoUrl: null,
      },
    };
    render(
      <MemoryRouter>
        <LiveNowSection games={[mlbLiveGame]} />
      </MemoryRouter>,
    );
    expect(
      screen.getByRole("link", { name: /Boston Red Sox/i }),
    ).toHaveAttribute("href", "/mlb/games/9");
  });

  it("renders team logos when logoUrl is set", () => {
    const { container } = render(
      <MemoryRouter>
        <LiveNowSection
          games={[
            {
              ...linkedLiveGame,
              away: {
                ...linkedLiveGame.away,
                logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
              },
              home: {
                ...linkedLiveGame.home,
                logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png",
              },
            },
          ]}
        />
      </MemoryRouter>,
    );
    const images = container.querySelectorAll("img");
    expect(images).toHaveLength(2);
    expect(images[0]).toHaveAttribute(
      "src",
      "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    );
  });
});
