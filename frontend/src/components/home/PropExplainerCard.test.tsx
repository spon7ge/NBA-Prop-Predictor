import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PropExplainerCard } from "./PropExplainerCard";

describe("PropExplainerCard", () => {
  it("renders player, line, model, FanDuel, and EV for Over", () => {
    render(
      <PropExplainerCard selectedSide="over" onSelectSide={vi.fn()} />,
    );
    expect(screen.getByText("LeBron James")).toBeInTheDocument();
    expect(screen.getByText("LAL · F")).toBeInTheDocument();
    expect(screen.getByText("DEN vs LAL")).toBeInTheDocument();
    expect(screen.getByText("Tue 7:00pm")).toBeInTheDocument();
    expect(screen.getByText("22.5")).toBeInTheDocument();
    expect(screen.getByText("Points")).toBeInTheDocument();
    expect(screen.getByText("24.7")).toBeInTheDocument();
    expect(screen.getByText("+4%")).toBeInTheDocument();
    expect(screen.getByText(/−110/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /over/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: /under/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("shows negative EV when Under is selected", () => {
    render(
      <PropExplainerCard selectedSide="under" onSelectSide={vi.fn()} />,
    );
    expect(screen.getByText("−4%")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /under/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("calls onSelectSide when toggling", async () => {
    const user = userEvent.setup();
    const onSelectSide = vi.fn();
    render(
      <PropExplainerCard selectedSide="over" onSelectSide={onSelectSide} />,
    );
    await user.click(screen.getByRole("button", { name: /under/i }));
    expect(onSelectSide).toHaveBeenCalledWith("under");
  });
});
