import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SeasonLeaders } from "./SeasonLeaders";
import { buildScheduledDetail } from "./testFixtures";

describe("SeasonLeaders", () => {
  it("renders points assists rebounds leaders", () => {
    render(
      <SeasonLeaders
        detail={buildScheduledDetail({
          seasonLeaders: {
            away: [
              {
                stat: "points",
                label: "Points",
                name: "Olivia Miles",
                value: "19.5",
              },
            ],
            home: [
              {
                stat: "points",
                label: "Points",
                name: "Marina Mabrey",
                value: "21.1",
              },
            ],
          },
        })}
      />,
    );
    expect(screen.getByText(/Season leaders/i)).toBeInTheDocument();
    expect(screen.getByText("Olivia Miles")).toBeInTheDocument();
    expect(screen.getByText("19.5")).toBeInTheDocument();
    expect(screen.getByText("Marina Mabrey")).toBeInTheDocument();
  });

  it("renders nothing without season leaders", () => {
    const { container } = render(
      <SeasonLeaders
        detail={buildScheduledDetail({ seasonLeaders: null })}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
