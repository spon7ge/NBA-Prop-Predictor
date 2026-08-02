import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ShotChart } from "./ShotChart";
import { detail } from "./testFixtures";

describe("ShotChart", () => {
  it("renders the shot chart title", () => {
    render(<ShotChart detail={detail} />);
    expect(screen.getByText("Shot chart")).toBeInTheDocument();
  });

  it("shows the FG line and data source", () => {
    render(<ShotChart detail={detail} />);
    expect(screen.getByText("1/2 FG")).toBeInTheDocument();
    expect(screen.getByText("Data: ESPN")).toBeInTheDocument();
  });

  it("shows the latest play text with period and clock", () => {
    render(<ShotChart detail={detail} />);
    expect(
      screen.getByText("Laeticia Amihere makes two point shot"),
    ).toBeInTheDocument();
    expect(screen.getByText("Q1 4:29")).toBeInTheDocument();
  });

  it("shows tip-off pending copy when there is no latest play", () => {
    render(<ShotChart detail={{ ...detail, latestPlay: null }} />);
    expect(screen.getByText(/tip-off pending/i)).toBeInTheDocument();
  });

  it("shows both teams' shots by default", () => {
    render(<ShotChart detail={detail} />);
    expect(screen.getByRole("img", { name: /A\. Player/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /B\. Player/ })).toBeInTheDocument();
  });

  it("filters shots to the selected team", async () => {
    const user = userEvent.setup();
    render(<ShotChart detail={detail} />);

    await user.click(screen.getByRole("button", { name: "GS" }));
    expect(screen.getByRole("img", { name: /A\. Player/ })).toBeInTheDocument();
    expect(
      screen.queryByRole("img", { name: /B\. Player/ }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "PHX" }));
    expect(
      screen.queryByRole("img", { name: /A\. Player/ }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("img", { name: /B\. Player/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Both" }));
    expect(screen.getByRole("img", { name: /A\. Player/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /B\. Player/ })).toBeInTheDocument();
  });

  it("filters shots by period under the chart", async () => {
    const user = userEvent.setup();
    const withPeriods = {
      ...detail,
      shots: [
        {
          id: "s1",
          teamId: detail.away.id,
          playerName: "A. Player",
          made: true,
          x: 25,
          y: 5,
          period: 1,
          clock: "8:00",
        },
        {
          id: "s2",
          teamId: detail.home.id,
          playerName: "B. Player",
          made: false,
          x: 20,
          y: 10,
          period: 2,
          clock: "7:00",
        },
        {
          id: "s3",
          teamId: detail.away.id,
          playerName: "C. Player",
          made: true,
          x: 30,
          y: 12,
          period: 3,
          clock: "6:00",
        },
      ],
      fgMade: 2,
      fgAttempted: 3,
    };
    render(<ShotChart detail={withPeriods} />);

    expect(screen.getByRole("button", { name: "All" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Q1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Q2" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Q3" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Q1" }));
    expect(screen.getByRole("img", { name: /A\. Player/ })).toBeInTheDocument();
    expect(
      screen.queryByRole("img", { name: /B\. Player/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("img", { name: /C\. Player/ }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("1/1 FG")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "All" }));
    expect(screen.getByRole("img", { name: /A\. Player/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /B\. Player/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /C\. Player/ })).toBeInTheDocument();
    expect(screen.getByText("2/3 FG")).toBeInTheDocument();
  });

  it("wraps content in the quiet GameSection surface", () => {
    render(<ShotChart detail={detail} />);
    const heading = screen.getByRole("heading", { name: /shot chart/i });
    expect(heading.closest("section")).toHaveClass(
      "rounded-xl",
      "border-white/10",
      "bg-white/[0.03]",
      "!p-3",
    );
  });
});
