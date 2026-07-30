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

  it("shows the latest play text", () => {
    render(<ShotChart detail={detail} />);
    expect(
      screen.getByText("Laeticia Amihere makes two point shot"),
    ).toBeInTheDocument();
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
});
