import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { WinProbabilityPanel } from "./WinProbabilityPanel";
import { buildGameDetailFixture } from "./testFixtures";
import type {
  GameDetailWinProbability,
  GameDetailWinProbabilityPoint,
} from "./types";

function buildDenseTimeline(count: number): GameDetailWinProbabilityPoint[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `wp-${index}`,
    period: 1,
    clock: `${Math.max(0, 10 - Math.floor(index / 40))}:${String(index % 60).padStart(2, "0")}`,
    awayScore: index,
    homeScore: Math.max(0, index - 1),
    awayWinPct: 50,
    homeWinPct: 50,
    teamId: "home1",
  }));
}

describe("WinProbabilityPanel", () => {
  it("renders the latest win probability point and team stats by default", () => {
    render(<WinProbabilityPanel detail={buildGameDetailFixture()} />);

    expect(screen.getByText("Win probability")).toBeInTheDocument();
    expect(
      screen.getByText("Above the midline favors PHX"),
    ).toBeInTheDocument();
    expect(screen.getByText("10-8")).toBeInTheDocument();
    expect(screen.getByText("PHX 54%")).toBeInTheDocument();
    expect(screen.getByText("GS 46%")).toBeInTheDocument();
    expect(screen.getByText("Field goal %")).toBeInTheDocument();
    expect(screen.getByText("49")).toBeInTheDocument();
    expect(screen.getByText("41")).toBeInTheDocument();
  });

  it("updates the active score snapshot when pointer moves near an earlier timeline point", () => {
    const { container } = render(
      <WinProbabilityPanel detail={buildGameDetailFixture()} />,
    );

    expect(screen.getByText("10-8")).toBeInTheDocument();
    expect(screen.getByText("PHX 54%")).toBeInTheDocument();

    const chart = container.querySelector("svg");
    expect(chart).not.toBeNull();
    chart!.getBoundingClientRect = () =>
      ({
        left: 0,
        top: 0,
        width: 320,
        height: 120,
        right: 320,
        bottom: 120,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      }) as DOMRect;

    // Two points map to x=0 and x=320; pointer near the left selects the earlier point.
    fireEvent.mouseMove(chart!, { clientX: 10, clientY: 60 });

    expect(screen.getByText("2-0")).toBeInTheDocument();
    expect(screen.getByText("PHX 44%")).toBeInTheDocument();
    expect(screen.getByText("GS 56%")).toBeInTheDocument();
    expect(screen.queryByText("10-8")).not.toBeInTheDocument();
  });

  it("lets keyboard users step away from the latest point via a single slider", async () => {
    const user = userEvent.setup();
    render(<WinProbabilityPanel detail={buildGameDetailFixture()} />);

    expect(screen.getByText("10-8")).toBeInTheDocument();

    const slider = screen.getByRole("slider", {
      name: /win probability timeline/i,
    });
    await user.tab();
    expect(slider).toHaveFocus();

    // jsdom does not implement native range arrow-key stepping; change still
    // proves a single focusable control can leave the default latest point.
    fireEvent.change(slider, { target: { value: "0" } });

    expect(screen.getByText("2-0")).toBeInTheDocument();
    expect(screen.getByText("PHX 44%")).toBeInTheDocument();
    expect(screen.getByText("GS 56%")).toBeInTheDocument();
  });
  it("does not create a focusable marker for every dense timeline point", () => {
    const winProbability: GameDetailWinProbability = {
      summary: null,
      timeline: buildDenseTimeline(400),
      teamStats: [],
    };

    render(
      <WinProbabilityPanel
        detail={buildGameDetailFixture({ winProbability })}
      />,
    );

    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(screen.getAllByRole("slider")).toHaveLength(1);
  });

  it("renders timeline-only data without team stats", () => {
    const winProbability: GameDetailWinProbability = {
      summary: null,
      timeline: [
        {
          id: "wp-1",
          period: 1,
          clock: "8:00",
          awayScore: 2,
          homeScore: 0,
          awayWinPct: 56,
          homeWinPct: 44,
          teamId: "away1",
        },
      ],
      teamStats: [],
    };

    render(
      <WinProbabilityPanel
        detail={buildGameDetailFixture({ winProbability })}
      />,
    );

    expect(screen.getByText("2-0")).toBeInTheDocument();
    expect(screen.getByText("PHX 44%")).toBeInTheDocument();
    expect(
      screen.getByRole("slider", { name: /win probability timeline/i }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Field goal %")).not.toBeInTheDocument();
  });

  it("renders stats-only data with away/home legend cues", () => {
    const winProbability: GameDetailWinProbability = {
      summary: null,
      timeline: [],
      teamStats: [
        {
          key: "field_goal_pct",
          label: "Field goal %",
          awayValue: 41,
          homeValue: 49,
        },
      ],
    };

    render(
      <WinProbabilityPanel
        detail={buildGameDetailFixture({ winProbability })}
      />,
    );

    expect(screen.getByText("Field goal %")).toBeInTheDocument();
    expect(screen.getByText("41")).toBeInTheDocument();
    expect(screen.getByText("49")).toBeInTheDocument();
    expect(screen.getByText("GS")).toBeInTheDocument();
    expect(screen.getByText("PHX")).toBeInTheDocument();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("shows an unavailable message when win probability data is missing", () => {
    render(
      <WinProbabilityPanel
        detail={buildGameDetailFixture({ winProbability: null })}
      />,
    );

    expect(
      screen.getByText("Win probability unavailable for this game yet."),
    ).toBeInTheDocument();
  });
});
