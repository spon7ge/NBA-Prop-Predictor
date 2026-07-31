import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BoxScore } from "./BoxScore";
import { buildGameDetailFixture } from "./testFixtures";

describe("BoxScore", () => {
  it("renders both team box scores with player stats and DNP", () => {
    render(
      <BoxScore
        detail={buildGameDetailFixture({
          boxScore: {
            columns: [
              "MIN",
              "PTS",
              "FG",
              "3PT",
              "FT",
              "REB",
              "AST",
              "TO",
              "STL",
              "BLK",
              "OREB",
              "DREB",
              "PF",
              "+/-",
            ],
            away: [
              {
                name: "Kayla Thornton",
                didNotPlay: false,
                values: [
                  "25",
                  "6",
                  "2-7",
                  "2-5",
                  "0-0",
                  "8",
                  "2",
                  "0",
                  "0",
                  "0",
                  "4",
                  "4",
                  "5",
                  "+6",
                ],
              },
              {
                name: "Gabby Williams",
                didNotPlay: true,
                values: Array(14).fill(""),
              },
            ],
            home: [
              {
                name: "Alyssa Thomas",
                didNotPlay: false,
                values: [
                  "30",
                  "12",
                  "5-10",
                  "0-1",
                  "2-2",
                  "7",
                  "5",
                  "2",
                  "1",
                  "0",
                  "2",
                  "5",
                  "3",
                  "+4",
                ],
              },
            ],
          },
        })}
      />,
    );

    expect(screen.getByText("GS")).toBeInTheDocument();
    expect(screen.getByText("Golden State Valkyries")).toBeInTheDocument();
    expect(screen.getByText("PHX")).toBeInTheDocument();
    expect(screen.getByText("Kayla Thornton")).toBeInTheDocument();
    expect(screen.getByText("Gabby Williams")).toBeInTheDocument();
    expect(screen.getByText("DNP")).toBeInTheDocument();
    expect(screen.getByText("Alyssa Thomas")).toBeInTheDocument();
    expect(screen.getAllByText("MIN").length).toBeGreaterThan(0);
    expect(screen.getAllByText("+/-").length).toBeGreaterThan(0);
  });

  it("renders nothing without box score data", () => {
    const { container } = render(
      <BoxScore detail={buildGameDetailFixture({ boxScore: null })} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
