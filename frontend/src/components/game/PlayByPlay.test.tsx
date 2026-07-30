import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PlayByPlay } from "./PlayByPlay";
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
});
