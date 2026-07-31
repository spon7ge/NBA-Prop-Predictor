import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { GameDetailPage } from "./GameDetailPage";

function renderGameDetail(espnEventId: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/games/${espnEventId}`]}>
        <Routes>
          <Route path="/games/:espnEventId" element={<GameDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function previewFields() {
  return {
    matchup_prediction: {
      away_win_pct: 67,
      home_win_pct: 33,
      source_label: "ESPN game projection",
    },
    projected_starters: {
      note: "from each team's last game",
      away: [{ jersey: "1", name: "Natasha Howard", position: "F" }],
      home: [{ jersey: "10", name: "Maria Conde", position: "F" }],
    },
    season_leaders: {
      away: [
        { stat: "points" as const, label: "Points", name: "Player A", value: "20.1" },
      ],
      home: [
        { stat: "points" as const, label: "Points", name: "Player B", value: "18.5" },
      ],
    },
    injuries: null,
  };
}

function baseGameDetail(overrides: Record<string, unknown> = {}) {
  return {
    espn_event_id: "401857099",
    league: "wnba",
    status: "scheduled",
    status_label: "Sun, July 30 at 7:00 PM EDT",
    venue: "Scotiabank Arena",
    away: {
      id: "129153",
      abbrev: "MIN",
      name: "Minnesota Lynx",
      score: null,
      color: "#266092",
      logo_url: null,
    },
    home: {
      id: "21",
      abbrev: "TOR",
      name: "Toronto Tempo",
      score: null,
      color: "#CE1141",
      logo_url: null,
    },
    fg_made: 0,
    fg_attempted: 0,
    latest_play: null,
    shots: [],
    plays: [],
    win_probability: null,
    ...previewFields(),
    fetched_at: "",
    ...overrides,
  };
}

describe("GameDetailPage", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows matchup preview sections for scheduled games", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => baseGameDetail(),
    });

    renderGameDetail("401857099");

    expect(await screen.findByText(/Matchup prediction/i)).toBeInTheDocument();
    expect(screen.getByText(/Projected starters/i)).toBeInTheDocument();
    expect(screen.getByText(/Season leaders/i)).toBeInTheDocument();
    expect(screen.queryByText(/Shot chart/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Play-by-play/i)).not.toBeInTheDocument();
  });

  it("shows live panels for live games", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () =>
        baseGameDetail({
          espn_event_id: "401857098",
          status: "live",
          status_label: "4:13 - 1st",
          away: {
            id: "129153",
            abbrev: "GS",
            name: "Golden State Valkyries",
            score: 10,
            color: "#553987",
            logo_url: null,
          },
          home: {
            id: "21",
            abbrev: "PHX",
            name: "Phoenix Mercury",
            score: 9,
            color: "#E56020",
            logo_url: null,
          },
          fg_made: 1,
          fg_attempted: 2,
          matchup_prediction: null,
          projected_starters: null,
          season_leaders: null,
          injuries: null,
        }),
    });

    renderGameDetail("401857098");

    expect(await screen.findByText(/Shot chart/i)).toBeInTheDocument();
    expect(screen.getByText(/Play-by-play/i)).toBeInTheDocument();
    expect(screen.queryByText(/Matchup prediction/i)).not.toBeInTheDocument();
  });
});
