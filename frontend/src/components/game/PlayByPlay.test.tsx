import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ApiWnbaGameDetail } from "@/lib/api";
import { PlayByPlay } from "./PlayByPlay";
import { mapGameDetail } from "./mapGameDetail";
import { detail } from "./testFixtures";

describe("PlayByPlay", () => {
  it("defaults to the latest period with plays", () => {
    render(<PlayByPlay detail={detail} />);
    expect(
      screen.getByText("B. Player makes three point shot"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("A. Player makes two point shot"),
    ).not.toBeInTheDocument();
  });

  it("filters plays when a period pill is clicked", async () => {
    const user = userEvent.setup();
    render(<PlayByPlay detail={detail} />);

    await user.click(screen.getByRole("button", { name: "1st" }));
    expect(
      screen.getByText("A. Player makes two point shot"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("B. Player makes three point shot"),
    ).not.toBeInTheDocument();
  });

  it("renders the newest play first for a period, matching API order", async () => {
    const user = userEvent.setup();
    render(<PlayByPlay detail={detail} />);

    await user.click(screen.getByRole("button", { name: "1st" }));
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("A. Player makes two point shot");
    expect(items[0]).toHaveTextContent("8:00");
    expect(items[1]).toHaveTextContent("Tip off won by Golden State");
    expect(items[1]).toHaveTextContent("10:00");
  });

  it("shows the newest play first when built from an API-shaped payload", () => {
    const apiDetail: ApiWnbaGameDetail = {
      espn_event_id: "401857098",
      league: "wnba",
      status: "live",
      status_label: "4:13 - 1st",
      venue: "Mortgage Matchup Center",
      away: { id: "away1", abbrev: "GS", name: "GS", score: 10, color: "#553987", logo_url: null },
      home: { id: "home1", abbrev: "PHX", name: "PHX", score: 9, color: "#E56020", logo_url: null },
      fg_made: 1,
      fg_attempted: 1,
      latest_play: null,
      shots: [],
      // API returns plays newest-first: the most recent play (smallest clock) comes first.
      plays: [
        {
          id: "pl2",
          team_id: "away1",
          period: 1,
          clock: "4:13",
          text: "Veronica Burton shooting foul",
          scoring: false,
          away_score: 10,
          home_score: 9,
          shooting: false,
        },
        {
          id: "pl1",
          team_id: "away1",
          period: 1,
          clock: "4:29",
          text: "Laeticia Amihere makes two point shot",
          scoring: true,
          away_score: 10,
          home_score: 8,
          shooting: true,
        },
      ],
      win_probability: null,
      matchup_prediction: null,
      projected_starters: null,
      season_leaders: null,
      injuries: null,
      box_score: null,
      fetched_at: "2026-07-29T19:00:00-04:00",
    };

    render(<PlayByPlay detail={mapGameDetail(apiDetail)} />);
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("Veronica Burton shooting foul");
    expect(items[1]).toHaveTextContent("Laeticia Amihere makes two point shot");
  });

  it("shows the running score on scoring plays", () => {
    render(<PlayByPlay detail={detail} />);
    expect(screen.getByText("2-3")).toBeInTheDocument();
  });

  it("does not show a score on non-scoring plays", async () => {
    const user = userEvent.setup();
    render(<PlayByPlay detail={detail} />);
    await user.click(screen.getByRole("button", { name: "1st" }));
    expect(screen.queryByText("0-0")).not.toBeInTheDocument();
  });

  it("shows a pending message when there are no plays yet", () => {
    render(<PlayByPlay detail={{ ...detail, plays: [] }} />);
    expect(screen.getByText(/tip-off pending/i)).toBeInTheDocument();
  });

  it("shows only the 10 newest plays for the active period", () => {
    const manyPlays = Array.from({ length: 12 }, (_, i) => ({
      id: `p${i}`,
      teamId: detail.away.id,
      period: 2,
      clock: `${9 - Math.floor(i / 2)}:${(50 - i).toString().padStart(2, "0")}`,
      text: `Play number ${i}`,
      scoring: false,
      awayScore: 0,
      homeScore: 0,
      shooting: false,
    }));
    render(<PlayByPlay detail={{ ...detail, plays: manyPlays }} />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(10);
    expect(items[0]).toHaveTextContent("Play number 0");
    expect(items[9]).toHaveTextContent("Play number 9");
    expect(screen.queryByText("Play number 10")).not.toBeInTheDocument();
    expect(screen.queryByText("Play number 11")).not.toBeInTheDocument();
  });

  it("wraps content in the quiet GameSection surface", () => {
    render(<PlayByPlay detail={detail} />);
    const heading = screen.getByRole("heading", { name: /play-by-play/i });
    expect(heading.closest("section")).toHaveClass(
      "rounded-xl",
      "border-white/10",
      "bg-white/[0.03]",
      "!p-3",
    );
  });

  it("renders scoring scores in white mono without amber", () => {
    const { container } = render(<PlayByPlay detail={detail} />);
    expect(container.querySelector(".text-amber-300")).toBeNull();
    const score = screen.getByText("2-3");
    expect(score.className).toMatch(/text-white/);
    expect(score.className).toMatch(/font-mono/);
  });
});
