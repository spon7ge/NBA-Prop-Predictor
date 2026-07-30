import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { InjuryReport } from "./InjuryReport";
import { buildScheduledDetail } from "./testFixtures";

describe("InjuryReport", () => {
  it("renders injury rows", () => {
    render(
      <InjuryReport
        detail={buildScheduledDetail({
          injuries: {
            away: [],
            home: [
              {
                name: "Nyara Sabally",
                position: "F",
                status: "Out",
                detail: "Ribs",
              },
            ],
          },
        })}
      />,
    );
    expect(screen.getByText(/Injury report/i)).toBeInTheDocument();
    expect(screen.getByText("Nyara Sabally")).toBeInTheDocument();
    expect(screen.getByText(/Out/)).toBeInTheDocument();
    expect(screen.getByText(/Ribs/)).toBeInTheDocument();
  });

  it("shows None listed for empty side when other side has injuries", () => {
    render(
      <InjuryReport
        detail={buildScheduledDetail({
          injuries: {
            away: [],
            home: [
              {
                name: "Nyara Sabally",
                position: "F",
                status: "Out",
                detail: "Ribs",
              },
            ],
          },
        })}
      />,
    );
    expect(screen.getByText("None listed")).toBeInTheDocument();
  });

  it("renders nothing without injuries", () => {
    const { container } = render(
      <InjuryReport detail={buildScheduledDetail({ injuries: null })} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
