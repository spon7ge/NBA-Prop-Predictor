import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MatchupPrediction } from "./MatchupPrediction";
import { buildScheduledDetail } from "./testFixtures";

describe("MatchupPrediction", () => {
  it("renders prediction bar and source", () => {
    render(
      <MatchupPrediction
        detail={buildScheduledDetail({
          matchupPrediction: {
            awayWinPct: 67,
            homeWinPct: 33,
            sourceLabel: "ESPN game projection",
          },
        })}
      />,
    );
    expect(screen.getByText(/Matchup prediction/i)).toBeInTheDocument();
    expect(screen.getByText(/MIN/)).toBeInTheDocument();
    expect(screen.getByText("67%")).toBeInTheDocument();
    expect(screen.getByText("33%")).toBeInTheDocument();
    expect(screen.getByText("ESPN game projection")).toBeInTheDocument();
  });

  it("renders nothing without prediction", () => {
    const { container } = render(
      <MatchupPrediction
        detail={buildScheduledDetail({ matchupPrediction: null })}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
